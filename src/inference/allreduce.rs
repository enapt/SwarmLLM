//! AllReduce for tensor-parallel inference.
//!
//! Two strategies:
//!
//! **Star topology** (current default, optimal for tp_size ≤ 3):
//! 1. Each TP rank sends its full partial to the coordinator (rank 0).
//! 2. Coordinator collects all partials, sums, broadcasts result.
//! 3. Bandwidth: coordinator sees 2N tensor transfers (bottleneck).
//!
//! **Ring topology** (optimal for tp_size ≥ 4, large tensors):
//! 1. Scatter-reduce: N-1 steps, each rank sends one chunk to right neighbor,
//!    receives from left, accumulates. After this, each rank holds one fully-reduced chunk.
//! 2. Allgather: N-1 steps, each rank sends its reduced chunk around the ring.
//!    After this, all ranks have the complete reduced tensor.
//! 3. Bandwidth: each node sends/receives 2*(N-1)/N of the tensor total.
//!    For N=4, that's 1.5x vs star's 2x at the coordinator.

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{
    AllReduceOp, NetworkCommand, TensorParallelGroup, TpAllReduceRequest, TpAllReduceResponse,
};

/// Timeout for AllReduce collection from all ranks.
const ALLREDUCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Minimum tensor size (f32 elements) to prefer ring over star.
const RING_MIN_TENSOR_ELEMENTS: usize = 1024;

/// Minimum TP group size to prefer ring over star.
const RING_MIN_TP_SIZE: u32 = 4;

// ─── Strategy selection ──────────────────────────────────────────────────────

/// Strategy for performing allreduce across TP ranks.
#[derive(Clone, Debug, PartialEq, Eq)]
enum AllReduceStrategy {
    /// Star: all ranks → coordinator → broadcast. Good for small groups.
    Star,
    /// Ring: scatter-reduce + allgather. Bandwidth-optimal for large groups.
    Ring,
}

/// Choose strategy based on group size and tensor size.
fn choose_allreduce_strategy(tp_size: u32, tensor_elements: usize) -> AllReduceStrategy {
    if tp_size >= RING_MIN_TP_SIZE && tensor_elements >= RING_MIN_TENSOR_ELEMENTS {
        AllReduceStrategy::Ring
    } else {
        AllReduceStrategy::Star
    }
}

// ─── Ring schedule types ─────────────────────────────────────────────────────

/// Phase of the ring allreduce algorithm.
#[derive(Clone, Debug, PartialEq, Eq)]
enum RingPhase {
    /// Scatter-reduce: accumulate partial sums in chunks around the ring.
    ScatterReduce,
    /// Allgather: propagate fully-reduced chunks around the ring.
    Allgather,
}

/// One communication step in the ring allreduce.
#[derive(Clone, Debug)]
struct RingAllReduceStep {
    /// Step index within the phase (0..n-1).
    pub step: usize,
    /// Phase.
    pub phase: RingPhase,
    /// Chunk index to send to right neighbor.
    pub send_chunk_idx: usize,
    /// Chunk index to receive from left neighbor.
    pub recv_chunk_idx: usize,
}

/// Compute the full ring schedule for a given rank in a group of `n` ranks.
/// Returns `2*(n-1)` steps: first `n-1` scatter-reduce, then `n-1` allgather.
fn compute_ring_schedule(rank: usize, n: usize) -> Vec<RingAllReduceStep> {
    if n <= 1 {
        return vec![];
    }
    let mut steps = Vec::with_capacity(2 * (n - 1));

    // Scatter-reduce phase
    for s in 0..(n - 1) {
        steps.push(RingAllReduceStep {
            step: s,
            phase: RingPhase::ScatterReduce,
            send_chunk_idx: (rank + n - s) % n,
            recv_chunk_idx: (rank + n - 1 - s) % n,
        });
    }

    // Allgather phase
    for s in 0..(n - 1) {
        steps.push(RingAllReduceStep {
            step: s,
            phase: RingPhase::Allgather,
            send_chunk_idx: (rank + n + 1 - s) % n,
            recv_chunk_idx: (rank + n - s) % n,
        });
    }

    steps
}

// ─── Ring allreduce (local simulation) ───────────────────────────────────────

/// Perform ring allreduce (sum) locally across N partial f32 tensors.
///
/// This simulates the full ring scatter-reduce + allgather algorithm.
/// For actual distributed execution, each step would be a network send/recv pair.
#[cfg(test)]
fn ring_allreduce_sum_local(partials: &[Vec<f32>]) -> Result<Vec<f32>, SwarmError> {
    let n = partials.len();
    if n == 0 {
        return Err(SwarmError::Internal("Ring allreduce: no partials".into()));
    }
    if n == 1 {
        return Ok(partials[0].clone());
    }
    let len = partials[0].len();
    for (i, p) in partials.iter().enumerate() {
        if p.len() != len {
            return Err(SwarmError::Internal(format!(
                "Ring allreduce: rank {i} has length {} but rank 0 has {len}",
                p.len()
            )));
        }
    }
    if len == 0 {
        return Ok(vec![]);
    }

    let chunk_size = len.div_ceil(n);

    // Split each rank's tensor into n chunks
    let mut rank_chunks: Vec<Vec<Vec<f32>>> = partials
        .iter()
        .map(|p| {
            (0..n)
                .map(|c| {
                    let start = c * chunk_size;
                    let end = (start + chunk_size).min(len);
                    if start < len {
                        p[start..end].to_vec()
                    } else {
                        vec![]
                    }
                })
                .collect()
        })
        .collect();

    // Scatter-reduce: N-1 steps
    for s in 0..(n - 1) {
        // Snapshot the chunks to send before mutating
        let send_chunks: Vec<Vec<f32>> = (0..n)
            .map(|rank| {
                let send_idx = (rank + n - s) % n;
                rank_chunks[rank][send_idx].clone()
            })
            .collect();

        for (rank, chunks) in rank_chunks.iter_mut().enumerate() {
            let recv_idx = (rank + n - 1 - s) % n;
            let left = (rank + n - 1) % n;
            let received = &send_chunks[left];
            for (j, val) in received.iter().enumerate() {
                if j < chunks[recv_idx].len() {
                    chunks[recv_idx][j] += val;
                }
            }
        }
    }

    // Allgather: N-1 steps
    for s in 0..(n - 1) {
        let send_chunks: Vec<Vec<f32>> = (0..n)
            .map(|rank| {
                let send_idx = (rank + n + 1 - s) % n;
                rank_chunks[rank][send_idx].clone()
            })
            .collect();

        for (rank, chunks) in rank_chunks.iter_mut().enumerate() {
            let recv_idx = (rank + n - s) % n;
            let left = (rank + n - 1) % n;
            chunks[recv_idx] = send_chunks[left].clone();
        }
    }

    // Reassemble from rank 0 (all ranks are identical now)
    let mut result = Vec::with_capacity(len);
    for chunk in &rank_chunks[0] {
        result.extend_from_slice(chunk);
    }
    result.truncate(len);
    Ok(result)
}

/// Ring allreduce over zstd-compressed partial tensors.
///
/// Drop-in replacement for `TpAllReduceCollector::reduce_sum()` using the ring algorithm.
/// Input: one compressed f32 tensor per rank, all with the same shape.
/// Output: compressed reduced tensor + shape.
#[cfg(test)]
fn ring_allreduce_sum_compressed(
    compressed_partials: &[Vec<u8>],
    shape: &[u32],
) -> Result<(Vec<u8>, Vec<u32>), SwarmError> {
    let elem_count: usize = shape
        .iter()
        .try_fold(1usize, |acc, &s| acc.checked_mul(s as usize))
        .ok_or_else(|| SwarmError::Internal("Ring allreduce: shape overflow".into()))?;

    if elem_count > 64 * 1024 * 1024 {
        return Err(SwarmError::Internal(
            "Ring allreduce: tensor too large".into(),
        ));
    }

    // Decompress each partial into Vec<f32>
    let partials: Vec<Vec<f32>> = compressed_partials
        .iter()
        .enumerate()
        .map(|(i, data)| {
            let max_decompressed = elem_count * 4 + 1024;
            let dec = {
                let mut decoder = zstd::Decoder::new(std::io::Cursor::new(data)).map_err(|e| {
                    SwarmError::Internal(format!("Ring allreduce: zstd init rank {i}: {e}"))
                })?;
                let mut buf = Vec::with_capacity(elem_count * 4);
                use std::io::Read;
                decoder
                    .by_ref()
                    .take(max_decompressed as u64)
                    .read_to_end(&mut buf)
                    .map_err(|e| {
                        SwarmError::Internal(format!("Ring allreduce: zstd rank {i}: {e}"))
                    })?;
                buf
            };
            if dec.len() != elem_count * 4 {
                return Err(SwarmError::Internal(format!(
                    "Ring allreduce: rank {i} size mismatch: {} vs expected {}",
                    dec.len(),
                    elem_count * 4
                )));
            }
            Ok(dec
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let reduced = ring_allreduce_sum_local(&partials)?;

    // Compress result
    let raw: Vec<u8> = reduced.iter().flat_map(|f| f.to_le_bytes()).collect();
    let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1)
        .map_err(|e| SwarmError::Internal(format!("Ring allreduce: zstd compress: {e}")))?;
    Ok((compressed, shape.to_vec()))
}

// ─── Existing infrastructure (unchanged) ─────────────────────────────────────

/// Pending AllReduce response channel, keyed by (request_id, layer_idx).
type PendingAllReduceMap = dashmap::DashMap<(Uuid, u32), oneshot::Sender<TpAllReduceResponse>>;

/// Shared registry of pending AllReduce responses.
/// Pipeline executors register here; the daemon dispatcher delivers responses.
pub struct AllReduceRegistry {
    pending: PendingAllReduceMap,
}

impl Default for AllReduceRegistry {
    fn default() -> Self {
        Self {
            pending: dashmap::DashMap::new(),
        }
    }
}

impl AllReduceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pending allreduce and return the receiver.
    pub fn register(
        &self,
        request_id: Uuid,
        layer_idx: u32,
    ) -> oneshot::Receiver<TpAllReduceResponse> {
        let (tx, rx) = oneshot::channel();
        self.pending.insert((request_id, layer_idx), tx);
        rx
    }

    /// Deliver a response to the waiting pipeline executor.
    /// Returns false if no one was waiting (timed out or duplicate).
    pub fn deliver(&self, resp: TpAllReduceResponse) -> bool {
        if let Some((_, tx)) = self.pending.remove(&(resp.request_id, resp.layer_idx)) {
            tx.send(resp).is_ok()
        } else {
            false
        }
    }

    /// Clean up stale entries where the receiver has been dropped (timed out).
    pub fn cleanup_stale(&self) {
        let stale_keys: Vec<(Uuid, u32)> = self
            .pending
            .iter()
            .filter(|entry| entry.value().is_closed())
            .map(|entry| *entry.key())
            .collect();
        if !stale_keys.is_empty() {
            tracing::debug!(
                count = stale_keys.len(),
                "Cleaning up stale AllReduce entries"
            );
            for key in stale_keys {
                self.pending.remove(&key);
            }
        }
    }
}

// ─── Ring chunk delivery registry ─────────────────────────────────────────────

/// Key for a pending ring chunk: (request_id, layer_idx, step).
type RingChunkKey = (Uuid, u32, u32);

/// Registry for receiving ring AllReduce chunks from neighbors.
pub struct RingChunkRegistry {
    pending: dashmap::DashMap<RingChunkKey, oneshot::Sender<Vec<u8>>>,
}

impl Default for RingChunkRegistry {
    fn default() -> Self {
        Self {
            pending: dashmap::DashMap::new(),
        }
    }
}

impl RingChunkRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register to receive a chunk for a specific (request, layer, step).
    pub fn register(
        &self,
        request_id: Uuid,
        layer_idx: u32,
        step: u32,
    ) -> oneshot::Receiver<Vec<u8>> {
        let (tx, rx) = oneshot::channel();
        self.pending.insert((request_id, layer_idx, step), tx);
        rx
    }

    /// Remove entries whose receiver has been dropped (timed-out operations).
    pub fn cleanup_stale(&self) {
        self.pending.retain(|_, tx| !tx.is_closed());
    }

    /// Deliver a received chunk. Returns false if no one was waiting.
    pub fn deliver(&self, request_id: Uuid, layer_idx: u32, step: u32, data: Vec<u8>) -> bool {
        if let Some((_, tx)) = self.pending.remove(&(request_id, layer_idx, step)) {
            tx.send(data).is_ok()
        } else {
            false
        }
    }
}

// ─── Ring AllReduce network execution ─────────────────────────────────────────

/// Execute ring AllReduce over the network for a single layer.
///
/// Each node holds one partial tensor. The ring algorithm runs 2*(N-1) steps:
/// - Scatter-reduce (N-1 steps): accumulate partial sums chunk-by-chunk around ring
/// - Allgather (N-1 steps): propagate fully-reduced chunks around ring
///
/// After completion, all nodes have the fully-reduced tensor.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn ring_allreduce_network(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    ring_registry: &RingChunkRegistry,
    allreduce_registry: &AllReduceRegistry,
    request_id: Uuid,
    layer_idx: u32,
    tp_group: &TensorParallelGroup,
    local_rank: usize,
    partial_data_compressed: Vec<u8>,
    shape: Vec<u32>,
) -> Result<TpAllReduceResponse, SwarmError> {
    let n = tp_group.tp_size();
    let schedule = compute_ring_schedule(local_rank, n);

    // Decompress our partial tensor to f32
    let partial_bytes = zstd::decode_all(std::io::Cursor::new(&partial_data_compressed))
        .map_err(|e| SwarmError::Internal(format!("Decompress ring partial: {e}")))?;
    if partial_bytes.len() % 4 != 0 {
        return Err(SwarmError::Internal(format!(
            "Ring partial data length {} not aligned to 4 bytes",
            partial_bytes.len()
        )));
    }
    let num_elements = partial_bytes.len() / 4;
    let mut local_data: Vec<f32> = vec![0.0; num_elements];
    for (i, chunk) in partial_bytes.chunks_exact(4).enumerate() {
        local_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Split into N chunks
    let chunk_size = num_elements.div_ceil(n);
    let mut chunks: Vec<Vec<f32>> = (0..n)
        .map(|c| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(num_elements);
            if start < num_elements {
                local_data[start..end].to_vec()
            } else {
                vec![]
            }
        })
        .collect();

    // Right neighbor to send to
    let right_rank = (local_rank + 1) % n;
    let right_node = &tp_group.nodes[right_rank];
    let right_peer_bytes = shared_state
        .peer_id_map
        .get(right_node)
        .map(|r| r.clone())
        .ok_or_else(|| {
            SwarmError::Internal(format!("No PeerId for ring right neighbor {right_node}"))
        })?;

    // Execute each step of the ring
    for ring_step in &schedule {
        let step_num = ring_step.step
            + if ring_step.phase == RingPhase::Allgather {
                n - 1
            } else {
                0
            };

        // Register to receive from left neighbor for this step
        let rx = ring_registry.register(request_id, layer_idx, step_num as u32);

        // Send our chunk to the right neighbor
        let send_data: Vec<u8> = chunks[ring_step.send_chunk_idx]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let compressed_chunk = zstd::encode_all(std::io::Cursor::new(&send_data), 1)
            .map_err(|e| SwarmError::Internal(format!("Compress ring chunk: {e}")))?;

        let chunk_msg = crate::types::TpRingChunk {
            request_id,
            layer_idx,
            step: step_num as u32,
            chunk_idx: ring_step.send_chunk_idx as u32,
            is_allgather: ring_step.phase == RingPhase::Allgather,
            chunk_data: compressed_chunk,
            num_chunks: n as u32,
            sender_peer_bytes: None,
        };
        network_tx
            .send(NetworkCommand::SendRingChunk {
                target_peer_bytes: right_peer_bytes.clone(),
                chunk: chunk_msg,
            })
            .await
            .map_err(|e| SwarmError::Internal(format!("Send ring chunk: {e}")))?;

        // Wait for chunk from left neighbor
        let received_data = tokio::time::timeout(ALLREDUCE_TIMEOUT, rx)
            .await
            .map_err(|_| {
                SwarmError::Internal(format!(
                    "Ring AllReduce timeout at step {step_num} for layer {layer_idx}"
                ))
            })?
            .map_err(|_| SwarmError::Internal("Ring chunk channel dropped".into()))?;

        // Decompress received chunk
        let recv_bytes = zstd::decode_all(std::io::Cursor::new(&received_data))
            .map_err(|e| SwarmError::Internal(format!("Decompress ring recv: {e}")))?;
        if recv_bytes.len() % 4 != 0 {
            return Err(SwarmError::Internal(format!(
                "Ring recv data length {} not aligned to 4 bytes",
                recv_bytes.len()
            )));
        }
        let recv_floats: Vec<f32> = recv_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // SEC: Reject NaN/Inf from peers — a single non-finite value in the ring
        // poisons the reduced tensor, propagates through every subsequent layer,
        // and corrupts every token of the request output (and any sessions sharing
        // the KV cache).
        if !recv_floats.iter().all(|f| f.is_finite()) {
            return Err(SwarmError::Internal(format!(
                "Ring AllReduce step {step_num}: received non-finite values from peer"
            )));
        }

        // Apply the received chunk
        let recv_idx = ring_step.recv_chunk_idx;
        match ring_step.phase {
            RingPhase::ScatterReduce => {
                // Accumulate: add received to our chunk
                for (j, val) in recv_floats.iter().enumerate() {
                    if j < chunks[recv_idx].len() {
                        chunks[recv_idx][j] += val;
                    }
                }
            }
            RingPhase::Allgather => {
                // Replace: overwrite our chunk with the fully-reduced one
                chunks[recv_idx] = recv_floats;
            }
        }
    }

    // Reassemble the fully-reduced tensor
    let mut result: Vec<f32> = Vec::with_capacity(num_elements);
    for chunk in &chunks {
        result.extend_from_slice(chunk);
    }
    result.truncate(num_elements);

    // Compress and package as TpAllReduceResponse
    let result_bytes: Vec<u8> = result.iter().flat_map(|f| f.to_le_bytes()).collect();
    let reduced_compressed = zstd::encode_all(std::io::Cursor::new(&result_bytes), 1)
        .map_err(|e| SwarmError::Internal(format!("Compress ring result: {e}")))?;

    let resp = TpAllReduceResponse {
        request_id,
        layer_idx,
        reduced_data: reduced_compressed,
        shape,
    };

    // Deliver to ourselves (the pipeline is waiting on the registry)
    allreduce_registry.deliver(resp.clone());

    Ok(resp)
}

/// Send this node's partial tensor and wait for the reduced result.
///
/// Automatically selects **star** or **ring** topology based on group size and tensor size.
/// - Star (default, tp_size < 4): all ranks → coordinator → broadcast
/// - Ring (tp_size ≥ 4, tensor ≥ 1024 elements): scatter-reduce + allgather via TpRingChunk
#[allow(clippy::too_many_arguments)]
pub async fn allreduce_sum(
    shared_state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    allreduce_registry: &AllReduceRegistry,
    request_id: Uuid,
    layer_idx: u32,
    tp_group: &TensorParallelGroup,
    local_rank: usize,
    partial_data_compressed: Vec<u8>,
    shape: Vec<u32>,
) -> Result<TpAllReduceResponse, SwarmError> {
    let tp_size = tp_group.tp_size() as u32;

    // Estimate tensor elements for strategy selection (compressed size / ~2 for FP16)
    let est_elements = partial_data_compressed.len() / 2;
    let strategy = choose_allreduce_strategy(tp_size, est_elements);

    if strategy == AllReduceStrategy::Ring {
        return ring_allreduce_network(
            shared_state,
            network_tx,
            &shared_state.ring_chunk_registry,
            allreduce_registry,
            request_id,
            layer_idx,
            tp_group,
            local_rank,
            partial_data_compressed,
            shape,
        )
        .await;
    }

    // Star topology (default for small groups)
    let coordinator = &tp_group.nodes[0];
    let is_coordinator = local_rank == 0;

    // Register to receive the reduced result
    let rx = allreduce_registry.register(request_id, layer_idx);

    let req = TpAllReduceRequest {
        request_id,
        layer_idx,
        tp_rank: local_rank as u32,
        tp_size,
        partial_data: partial_data_compressed,
        shape,
        op: AllReduceOp::Sum,
        sender_peer_bytes: None,
    };

    if is_coordinator {
        // Insert directly into the collector (we are rank 0)
        let all_arrived = {
            let mut entry = shared_state
                .pending_tp_partials
                .entry((request_id, layer_idx))
                .or_insert_with(|| crate::daemon::TpAllReduceCollector::new(tp_size));
            entry.insert(req, None)
        };

        if all_arrived {
            // We were the last to arrive (edge case: tp_size=1)
            let collector = shared_state
                .pending_tp_partials
                .remove(&(request_id, layer_idx));
            if let Some((_, collector)) = collector {
                let (reduced_data, shape) = collector.reduce_sum()?;
                let resp = TpAllReduceResponse {
                    request_id,
                    layer_idx,
                    reduced_data,
                    shape,
                };
                // Deliver to ourselves
                allreduce_registry.deliver(resp.clone());
            }
        }
    } else {
        // Send partial to coordinator via point-to-point
        let peer_bytes = shared_state.peer_id_map.get(coordinator).map(|r| r.clone());
        if let Some(target_peer_bytes) = peer_bytes {
            network_tx
                .send(NetworkCommand::SendAllReduceRequest {
                    target_peer_bytes,
                    request: req,
                })
                .await
                .map_err(|e| SwarmError::Internal(format!("Send allreduce partial: {e}")))?;
        } else {
            return Err(SwarmError::Internal(format!(
                "No PeerId mapping for coordinator {}",
                coordinator
            )));
        }
    }

    // Wait for the reduced result
    match tokio::time::timeout(ALLREDUCE_TIMEOUT, rx).await {
        Ok(Ok(resp)) => Ok(resp),
        Ok(Err(_)) => Err(SwarmError::Internal(
            "AllReduce response channel dropped".into(),
        )),
        Err(_) => {
            // Cleanup stale entry
            allreduce_registry.pending.remove(&(request_id, layer_idx));
            shared_state
                .pending_tp_partials
                .remove(&(request_id, layer_idx));
            Err(SwarmError::Internal(format!(
                "AllReduce timeout after {}s for layer {}",
                ALLREDUCE_TIMEOUT.as_secs(),
                layer_idx
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::TpAllReduceCollector;

    // ── Helper ───────────────────────────────────────────────────────────

    fn compress_f32(vals: &[f32]) -> Vec<u8> {
        let raw: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        zstd::encode_all(std::io::Cursor::new(&raw), 1).unwrap()
    }

    fn decompress_f32(data: &[u8]) -> Vec<f32> {
        let dec = zstd::decode_all(std::io::Cursor::new(data)).unwrap();
        dec.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ── Existing star topology tests ─────────────────────────────────────

    #[test]
    fn test_allreduce_collector_single_rank() {
        let mut collector = TpAllReduceCollector::new(1);
        let req = TpAllReduceRequest {
            request_id: Uuid::new_v4(),
            layer_idx: 0,
            tp_rank: 0,
            tp_size: 1,
            partial_data: compress_f32(&[1.0, 2.0, 3.0]),
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
            sender_peer_bytes: None,
        };
        assert!(collector.insert(req, None));
        let (reduced, shape) = collector.reduce_sum().unwrap();
        assert_eq!(shape, vec![1, 1, 3]);
        assert_eq!(decompress_f32(&reduced), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_allreduce_collector_two_ranks() {
        let mut collector = TpAllReduceCollector::new(2);
        let request_id = Uuid::new_v4();

        let req0 = TpAllReduceRequest {
            request_id,
            layer_idx: 5,
            tp_rank: 0,
            tp_size: 2,
            partial_data: compress_f32(&[1.0, 2.0, 3.0]),
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
            sender_peer_bytes: None,
        };
        let req1 = TpAllReduceRequest {
            request_id,
            layer_idx: 5,
            tp_rank: 1,
            tp_size: 2,
            partial_data: compress_f32(&[4.0, 5.0, 6.0]),
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
            sender_peer_bytes: None,
        };

        assert!(!collector.insert(req0, None));
        assert!(collector.insert(req1, None));
        let (reduced, shape) = collector.reduce_sum().unwrap();
        assert_eq!(shape, vec![1, 1, 3]);
        assert_eq!(decompress_f32(&reduced), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_allreduce_registry() {
        let registry = AllReduceRegistry::new();
        let rid = Uuid::new_v4();
        let mut rx = registry.register(rid, 3);

        let resp = TpAllReduceResponse {
            request_id: rid,
            layer_idx: 3,
            reduced_data: vec![1, 2, 3],
            shape: vec![1],
        };

        assert!(registry.deliver(resp.clone()));
        let received = rx.try_recv().unwrap();
        assert_eq!(received.layer_idx, 3);
        assert!(!registry.deliver(resp));
    }

    #[test]
    fn test_allreduce_collector_four_ranks() {
        let mut collector = TpAllReduceCollector::new(4);
        let request_id = Uuid::new_v4();

        for rank in 0..4u32 {
            let val = rank as f32;
            let req = TpAllReduceRequest {
                request_id,
                layer_idx: 0,
                tp_rank: rank,
                tp_size: 4,
                partial_data: compress_f32(&[val, val, val]),
                shape: vec![1, 1, 3],
                op: AllReduceOp::Sum,
                sender_peer_bytes: None,
            };
            let all = collector.insert(req, None);
            assert_eq!(all, rank == 3);
        }

        let (reduced, _) = collector.reduce_sum().unwrap();
        assert_eq!(decompress_f32(&reduced), vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_allreduce_collector_out_of_order() {
        let mut collector = TpAllReduceCollector::new(3);
        let request_id = Uuid::new_v4();

        for &rank in &[2u32, 0, 1] {
            let val = (rank + 1) as f32;
            let req = TpAllReduceRequest {
                request_id,
                layer_idx: 7,
                tp_rank: rank,
                tp_size: 3,
                partial_data: compress_f32(&[val]),
                shape: vec![1, 1, 1],
                op: AllReduceOp::Sum,
                sender_peer_bytes: None,
            };
            collector.insert(req, None);
        }

        let (reduced, _) = collector.reduce_sum().unwrap();
        assert_eq!(decompress_f32(&reduced), vec![6.0]);
    }

    #[test]
    fn test_registry_undelivered() {
        let registry = AllReduceRegistry::new();
        let rid = Uuid::new_v4();
        let resp = TpAllReduceResponse {
            request_id: rid,
            layer_idx: 0,
            reduced_data: vec![],
            shape: vec![],
        };
        assert!(!registry.deliver(resp));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /// Naive element-wise sum of N f32 tensors. Reference for ring correctness.
    fn naive_sum(partials: &[Vec<f32>]) -> Vec<f32> {
        let len = partials[0].len();
        let mut result = vec![0.0f32; len];
        for p in partials {
            for (i, &v) in p.iter().enumerate() {
                result[i] += v;
            }
        }
        result
    }

    // ── Ring allreduce tests ─────────────────────────────────────────────

    #[test]
    fn test_ring_allreduce_two_ranks() {
        let partials = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let result = ring_allreduce_sum_local(&partials).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_ring_allreduce_three_ranks() {
        let partials = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
        ];
        let result = ring_allreduce_sum_local(&partials).unwrap();
        assert_eq!(result, vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_ring_allreduce_four_ranks() {
        let partials: Vec<Vec<f32>> = (0..4).map(|r| vec![r as f32; 4]).collect();
        let result = ring_allreduce_sum_local(&partials).unwrap();
        // 0+1+2+3 = 6
        assert_eq!(result, vec![6.0; 4]);
    }

    #[test]
    fn test_ring_allreduce_eight_ranks() {
        let partials: Vec<Vec<f32>> = (0..8).map(|r| vec![r as f32; 16]).collect();
        let result = ring_allreduce_sum_local(&partials).unwrap();
        // 0+1+2+3+4+5+6+7 = 28
        assert_eq!(result, vec![28.0; 16]);
    }

    #[test]
    fn test_ring_allreduce_non_divisible_length() {
        // 3 ranks, 5 elements (not divisible by 3 → chunks of size 2, last chunk size 1)
        let partials = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ];
        let result = ring_allreduce_sum_local(&partials).unwrap();
        assert_eq!(result, vec![111.0, 222.0, 333.0, 444.0, 555.0]);
    }

    #[test]
    fn test_ring_allreduce_single_rank() {
        let partials = vec![vec![42.0, 99.0]];
        let result = ring_allreduce_sum_local(&partials).unwrap();
        assert_eq!(result, vec![42.0, 99.0]);
    }

    #[test]
    fn test_ring_allreduce_compressed() {
        let shape = vec![1u32, 1, 4];
        let compressed: Vec<Vec<u8>> = vec![
            compress_f32(&[1.0, 2.0, 3.0, 4.0]),
            compress_f32(&[10.0, 20.0, 30.0, 40.0]),
            compress_f32(&[100.0, 200.0, 300.0, 400.0]),
            compress_f32(&[1000.0, 2000.0, 3000.0, 4000.0]),
        ];
        let (reduced, out_shape) = ring_allreduce_sum_compressed(&compressed, &shape).unwrap();
        assert_eq!(out_shape, shape);
        assert_eq!(
            decompress_f32(&reduced),
            vec![1111.0, 2222.0, 3333.0, 4444.0]
        );
    }

    #[test]
    fn test_choose_strategy() {
        assert_eq!(choose_allreduce_strategy(2, 4096), AllReduceStrategy::Star);
        assert_eq!(choose_allreduce_strategy(4, 4096), AllReduceStrategy::Ring);
        assert_eq!(choose_allreduce_strategy(4, 512), AllReduceStrategy::Star);
        assert_eq!(choose_allreduce_strategy(8, 8192), AllReduceStrategy::Ring);
        assert_eq!(
            choose_allreduce_strategy(3, 100_000),
            AllReduceStrategy::Star
        );
    }

    #[test]
    fn test_ring_schedule_four_ranks() {
        let schedule = compute_ring_schedule(0, 4);
        assert_eq!(schedule.len(), 6); // 2*(4-1)

        // First 3 steps are ScatterReduce
        for step in &schedule[..3] {
            assert_eq!(step.phase, RingPhase::ScatterReduce);
        }
        // Last 3 steps are Allgather
        for step in &schedule[3..] {
            assert_eq!(step.phase, RingPhase::Allgather);
        }

        // Verify scatter-reduce send/recv indices for rank 0 in N=4:
        // step 0: send chunk (0+4-0)%4=0, recv chunk (0+4-1-0)%4=3
        // step 1: send chunk (0+4-1)%4=3, recv chunk (0+4-1-1)%4=2
        // step 2: send chunk (0+4-2)%4=2, recv chunk (0+4-1-2)%4=1
        assert_eq!(schedule[0].send_chunk_idx, 0);
        assert_eq!(schedule[0].recv_chunk_idx, 3);
        assert_eq!(schedule[1].send_chunk_idx, 3);
        assert_eq!(schedule[1].recv_chunk_idx, 2);
        assert_eq!(schedule[2].send_chunk_idx, 2);
        assert_eq!(schedule[2].recv_chunk_idx, 1);
    }

    #[test]
    fn test_ring_schedule_empty_for_single() {
        assert!(compute_ring_schedule(0, 1).is_empty());
    }

    #[test]
    fn test_ring_matches_star() {
        // Verify ring produces identical results to naive sum for various configurations
        for n in [2, 3, 4, 5, 8] {
            for len in [1, 3, 7, 16, 100] {
                let partials: Vec<Vec<f32>> = (0..n)
                    .map(|r| (0..len).map(|i| r as f32 * 1000.0 + i as f32).collect())
                    .collect();

                let ring_result = ring_allreduce_sum_local(&partials).unwrap();
                let naive_result = naive_sum(&partials);

                assert_eq!(
                    ring_result.len(),
                    naive_result.len(),
                    "Length mismatch for n={n}, len={len}"
                );
                for (j, (r, n_val)) in ring_result.iter().zip(naive_result.iter()).enumerate() {
                    assert!(
                        (r - n_val).abs() < 1e-3,
                        "Mismatch at index {j} for n={n}, len={len}: ring={r}, naive={n_val}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_ring_allreduce_large_tensor() {
        // 4 ranks, 4096 elements each (realistic hidden dimension)
        let partials: Vec<Vec<f32>> = (0..4)
            .map(|r| (0..4096).map(|i| (r * 4096 + i) as f32 * 0.001).collect())
            .collect();

        let ring_result = ring_allreduce_sum_local(&partials).unwrap();
        let naive_result = naive_sum(&partials);
        assert_eq!(ring_result.len(), 4096);

        for (j, (r, n_val)) in ring_result.iter().zip(naive_result.iter()).enumerate() {
            assert!(
                (r - n_val).abs() < 1e-1,
                "Mismatch at {j}: ring={r}, naive={n_val}"
            );
        }
    }
}
