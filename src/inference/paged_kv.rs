//! PagedAttention KV cache: block-based memory pool for concurrent decode sessions.
//!
//! Instead of pre-allocating a dense KV buffer per request, all requests share a
//! pool of fixed-size blocks (BLOCK_SIZE tokens each). This eliminates per-session
//! memory waste and allows higher concurrency.
//!
//! **Architecture**:
//! - `PagedKvPool` — owns the physical GPU memory (key/value cache tensors) and a free list.
//! - `PagedKvEntry` — per-request metadata: logical→physical block mapping + seq length.
//! - `PagedKvStore` — concurrent map of active entries, keyed by (model_key, request_id).
//!
//! **CUDA-only**: The paged attention kernels (from vendored `candle-paged-attention`) require
//! CUDA. On CPU, callers should fall back to `candle_nn::kv_cache::KvCache` (Phase 1).
//!
//! **Feature gate**: `paged-attn` — requires `candle-cuda`.

use std::collections::{HashSet, VecDeque};
use std::sync::Mutex;

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use dashmap::DashMap;

/// Tokens per physical block.
pub const BLOCK_SIZE: usize = 16;

/// Physical block pool for paged KV cache.
///
/// Owns two large contiguous tensors (key_cache, value_cache) divided into fixed-size blocks.
/// Blocks are allocated from a free list and returned on request completion.
pub struct PagedKvPool {
    /// Key cache: `[num_blocks, n_kv_heads, head_dim/x, BLOCK_SIZE, x]`
    /// where `x` = element size in bytes (for packed storage).
    pub key_cache: Tensor,
    /// Value cache: `[num_blocks, n_kv_heads, head_dim, BLOCK_SIZE]`
    pub value_cache: Tensor,
    /// Free block indices (queue for FIFO allocation).
    free_blocks: Mutex<VecDeque<i32>>,
    /// Set of currently-free block IDs for O(1) double-free detection.
    free_set: Mutex<HashSet<i32>>,
    /// Total number of blocks in the pool.
    pub num_blocks: usize,
    /// Number of KV heads.
    pub n_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Element packing factor (x = sizeof(dtype) for key cache packed layout).
    pub x_factor: usize,
}

impl PagedKvPool {
    /// Create a new block pool on the given device.
    ///
    /// # Arguments
    /// - `num_blocks` — number of physical blocks to allocate.
    /// - `n_kv_heads` — number of key/value heads (after GQA grouping).
    /// - `head_dim` — dimension per head.
    /// - `dtype` — data type for cache tensors (F16, BF16, or F32).
    /// - `device` — CUDA device.
    pub fn new(
        num_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let x = match dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            _ => 4, // default
        };

        // Key cache: [num_blocks, n_kv_heads, head_dim/x, BLOCK_SIZE, x]
        assert_eq!(
            head_dim % x,
            0,
            "head_dim ({head_dim}) must be divisible by element packing factor ({x})"
        );
        let key_cache = Tensor::zeros(
            (num_blocks, n_kv_heads, head_dim / x, BLOCK_SIZE, x),
            dtype,
            device,
        )?;

        // Value cache: [num_blocks, n_kv_heads, head_dim, BLOCK_SIZE]
        let value_cache = Tensor::zeros(
            (num_blocks, n_kv_heads, head_dim, BLOCK_SIZE),
            dtype,
            device,
        )?;

        let free_blocks: VecDeque<i32> = (0..num_blocks as i32).collect();
        let free_set: HashSet<i32> = (0..num_blocks as i32).collect();

        Ok(Self {
            key_cache,
            value_cache,
            free_blocks: Mutex::new(free_blocks),
            free_set: Mutex::new(free_set),
            num_blocks,
            n_kv_heads,
            head_dim,
            x_factor: x,
        })
    }

    /// Allocate `count` blocks from the free list. Returns None if insufficient blocks.
    pub fn allocate(&self, count: usize) -> Option<Vec<i32>> {
        let mut free = self.free_blocks.lock().unwrap_or_else(|e| e.into_inner());
        if free.len() < count {
            tracing::debug!(
                requested = count,
                free_blocks = free.len(),
                "DIAG: paged_kv allocate FAILED — insufficient blocks"
            );
            return None;
        }
        let mut fset = self.free_set.lock().unwrap_or_else(|e| e.into_inner());
        let mut blocks = Vec::with_capacity(count);
        for _ in 0..count {
            let b = free.pop_front().expect("length checked above");
            fset.remove(&b);
            blocks.push(b);
        }
        tracing::debug!(
            blocks_allocated = count,
            free_blocks = free.len(),
            "DIAG: paged_kv allocate"
        );
        Some(blocks)
    }

    /// Return blocks to the free list.
    pub fn free(&self, blocks: &[i32]) {
        let mut free = self.free_blocks.lock().unwrap_or_else(|e| e.into_inner());
        let mut fset = self.free_set.lock().unwrap_or_else(|e| e.into_inner());
        for &b in blocks {
            if fset.insert(b) {
                // Only push if not already free (O(1) double-free detection)
                free.push_back(b);
            }
        }
    }

    /// Number of currently free blocks.
    pub fn free_count(&self) -> usize {
        self.free_blocks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Auto-size the number of blocks based on available VRAM budget.
    ///
    /// Returns the number of blocks that fit in `budget_mb` for the given model dimensions.
    pub fn auto_size(budget_mb: u64, n_kv_heads: usize, head_dim: usize, dtype: DType) -> usize {
        let bytes_per_elem = match dtype {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            _ => 4,
        };
        // Each block stores: key + value = 2 * n_kv_heads * head_dim * BLOCK_SIZE * bytes_per_elem
        let bytes_per_block = 2 * n_kv_heads * head_dim * BLOCK_SIZE * bytes_per_elem;
        if bytes_per_block == 0 {
            return 0;
        }
        let budget_bytes = budget_mb as usize * 1024 * 1024;
        budget_bytes / bytes_per_block
    }
}

/// Per-request paged KV cache metadata.
#[derive(Clone, Debug)]
pub struct PagedKvEntry {
    /// Logical block index → physical block ID mapping.
    pub block_table: Vec<i32>,
    /// Current sequence length (number of tokens cached).
    pub seq_len: usize,
    /// Last access time for TTL cleanup.
    pub last_accessed: std::time::Instant,
}

impl PagedKvEntry {
    /// Create a new entry with the given initial blocks.
    pub fn new(initial_blocks: Vec<i32>) -> Self {
        Self {
            block_table: initial_blocks,
            seq_len: 0,
            last_accessed: std::time::Instant::now(),
        }
    }

    /// Number of token slots available in current block allocation.
    pub fn capacity(&self) -> usize {
        self.block_table.len() * BLOCK_SIZE
    }

    /// Whether a new block is needed to store the next token.
    pub fn needs_new_block(&self) -> bool {
        self.seq_len >= self.capacity()
    }

    /// Compute the slot mapping for new tokens being appended.
    /// Returns a vector of absolute slot indices (block_id * BLOCK_SIZE + offset_within_block).
    pub fn slot_mapping_for_append(&self, num_new_tokens: usize) -> Vec<i64> {
        let mut slots = Vec::with_capacity(num_new_tokens);
        for i in 0..num_new_tokens {
            let pos = self.seq_len + i;
            let block_idx = pos / BLOCK_SIZE;
            let offset = pos % BLOCK_SIZE;
            if block_idx < self.block_table.len() {
                let physical_block = self.block_table[block_idx];
                slots.push((physical_block as usize * BLOCK_SIZE + offset) as i64);
            }
        }
        slots
    }
}

/// Concurrent store for per-request paged KV entries.
pub struct PagedKvStore {
    /// Active entries: (model_key, request_id) → entry.
    entries: DashMap<(String, String), PagedKvEntry>,
    /// TTL for abandoned entries.
    ttl: std::time::Duration,
}

impl PagedKvStore {
    /// Create a new store with the given TTL.
    pub fn new(ttl: std::time::Duration) -> Self {
        Self {
            entries: DashMap::new(),
            ttl,
        }
    }

    /// Get an existing entry, or return None.
    pub fn get(
        &self,
        model_key: &str,
        request_id: &str,
    ) -> Option<dashmap::mapref::one::RefMut<'_, (String, String), PagedKvEntry>> {
        let key = (model_key.to_string(), request_id.to_string());
        self.entries.get_mut(&key)
    }

    /// Insert a new entry.
    pub fn insert(&self, model_key: &str, request_id: &str, entry: PagedKvEntry) {
        let key = (model_key.to_string(), request_id.to_string());
        self.entries.insert(key, entry);
    }

    /// Remove an entry and return the block table for freeing.
    pub fn remove(&self, model_key: &str, request_id: &str) -> Option<Vec<i32>> {
        let key = (model_key.to_string(), request_id.to_string());
        self.entries
            .remove(&key)
            .map(|(_, entry)| entry.block_table)
    }

    /// Clean up expired entries. Returns block tables that should be freed.
    pub fn cleanup_expired(&self) -> Vec<Vec<i32>> {
        let ttl = self.ttl;
        let mut freed = Vec::new();
        self.entries.retain(|_, entry| {
            if entry.last_accessed.elapsed() > ttl {
                freed.push(entry.block_table.clone());
                false
            } else {
                true
            }
        });
        freed
    }

    /// Number of active entries.
    pub fn active_entries(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_kv_pool_allocate_free() {
        let pool = PagedKvPool::new(8, 2, 64, DType::F32, &Device::Cpu).unwrap();

        assert_eq!(pool.free_count(), 8);

        // Allocate 3 blocks
        let blocks = pool.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(pool.free_count(), 5);

        // Allocate more than available fails
        assert!(pool.allocate(6).is_none());

        // Free returns blocks
        pool.free(&blocks);
        assert_eq!(pool.free_count(), 8);
    }

    #[test]
    fn paged_kv_pool_auto_size() {
        // 2 KV heads, 128 head_dim, F16 (2 bytes)
        // Bytes per block = 2 * 2 * 128 * 16 * 2 = 16384 = 16 KB
        // 1 MB budget = 1048576 / 16384 = 64 blocks
        let blocks = PagedKvPool::auto_size(1, 2, 128, DType::F16);
        assert_eq!(blocks, 64);
    }

    #[test]
    fn block_table_grows_on_decode() {
        let pool = PagedKvPool::new(16, 2, 64, DType::F32, &Device::Cpu).unwrap();

        // Start with 1 block (16 token capacity)
        let blocks = pool.allocate(1).unwrap();
        let mut entry = PagedKvEntry::new(blocks);

        // Simulate 40+ tokens across 3 blocks
        for token in 0..40 {
            if entry.needs_new_block() {
                let new_block = pool.allocate(1).unwrap();
                entry.block_table.extend(new_block);
            }
            let slots = entry.slot_mapping_for_append(1);
            assert_eq!(slots.len(), 1);
            entry.seq_len += 1;

            // After adding tokens, verify we haven't overflowed
            assert!(
                entry.seq_len <= entry.capacity(),
                "seq_len {} > capacity {} at token {}",
                entry.seq_len,
                entry.capacity(),
                token
            );
        }

        // 40 tokens should need ceil(40/16) = 3 blocks
        assert_eq!(entry.block_table.len(), 3);
        assert_eq!(entry.seq_len, 40);
    }

    #[test]
    fn slot_mapping_correctness() {
        let mut entry = PagedKvEntry::new(vec![5, 10]); // 2 blocks: physical 5, 10
        entry.seq_len = 0;

        // First token goes to block 5, offset 0 → slot 80
        let slots = entry.slot_mapping_for_append(1);
        assert_eq!(slots, vec![5 * BLOCK_SIZE as i64]);

        entry.seq_len = 15;
        // Token at pos 15 goes to block 5, offset 15 → slot 95
        let slots = entry.slot_mapping_for_append(1);
        assert_eq!(slots, vec![5 * BLOCK_SIZE as i64 + 15]);

        entry.seq_len = 16;
        // Token at pos 16 goes to block 10, offset 0 → slot 160
        let slots = entry.slot_mapping_for_append(1);
        assert_eq!(slots, vec![10 * BLOCK_SIZE as i64]);
    }

    #[test]
    fn paged_kv_store_lifecycle() {
        let store = PagedKvStore::new(std::time::Duration::from_secs(600));

        // Insert an entry
        let entry = PagedKvEntry::new(vec![0, 1, 2]);
        store.insert("model-1", "req-a", entry);
        assert_eq!(store.active_entries(), 1);

        // Get and modify
        {
            let mut e = store.get("model-1", "req-a").unwrap();
            e.seq_len = 10;
            e.last_accessed = std::time::Instant::now();
        }

        // Remove returns block table
        let blocks = store.remove("model-1", "req-a").unwrap();
        assert_eq!(blocks, vec![0, 1, 2]);
        assert_eq!(store.active_entries(), 0);
    }

    #[test]
    fn paged_kv_store_cleanup_expired() {
        let store = PagedKvStore::new(std::time::Duration::from_millis(1));

        store.insert("model", "req-1", PagedKvEntry::new(vec![0, 1]));
        store.insert("model", "req-2", PagedKvEntry::new(vec![2, 3]));
        assert_eq!(store.active_entries(), 2);

        std::thread::sleep(std::time::Duration::from_millis(10));

        let freed = store.cleanup_expired();
        assert_eq!(freed.len(), 2);
        assert_eq!(store.active_entries(), 0);
    }
}
