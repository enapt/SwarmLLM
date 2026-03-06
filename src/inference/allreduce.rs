//! AllReduce coordinator for tensor-parallel inference.
//!
//! When multiple nodes compute partial results for the same layer,
//! the AllReduce coordinator collects all partials and sums them.
//!
//! **Flow (star topology, coordinator = rank-0 node):**
//! 1. Each TP rank computes its partial output for one layer.
//! 2. Each rank sends `TpAllReduceRequest` to the coordinator (rank 0).
//! 3. Coordinator collects all partials, sums, and broadcasts `TpAllReduceResponse`.
//! 4. Each rank receives the reduced tensor and continues to the next layer.
//!
//! Ring-AllReduce optimization (C6) replaces this with bandwidth-optimal
//! scatter-reduce + allgather for larger tensors.

use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::types::{AllReduceOp, NetworkCommand, TpAllReduceRequest, TpAllReduceResponse, TensorParallelGroup};

/// Timeout for AllReduce collection from all ranks.
const ALLREDUCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Pending AllReduce response channel, keyed by (request_id, layer_idx).
/// The pipeline executor registers this before sending its partial,
/// and the daemon dispatcher fires it when the reduced result arrives.
type PendingAllReduceMap =
    dashmap::DashMap<(Uuid, u32), oneshot::Sender<TpAllReduceResponse>>;

/// Shared registry of pending AllReduce responses.
/// Pipeline executors register here; the daemon dispatcher delivers responses.
pub struct AllReduceRegistry {
    pending: PendingAllReduceMap,
}

impl AllReduceRegistry {
    pub fn new() -> Self {
        Self {
            pending: dashmap::DashMap::new(),
        }
    }

    /// Register a pending allreduce and return the receiver.
    pub fn register(&self, request_id: Uuid, layer_idx: u32) -> oneshot::Receiver<TpAllReduceResponse> {
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

    /// Clean up stale entries older than the given duration.
    pub fn cleanup_stale(&self) {
        // Entries are removed on delivery or timeout; no-op for now.
        // Could add timestamps if needed.
    }
}

/// Send this node's partial tensor to the coordinator (rank 0) and wait for the reduced result.
///
/// If we ARE rank 0, the partial is inserted directly into the collector and we wait
/// for all other ranks to arrive (handled by the daemon dispatcher).
///
/// If we are NOT rank 0, we send the partial via the network and wait for the
/// broadcast response.
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
            let collector = shared_state.pending_tp_partials.remove(&(request_id, layer_idx));
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
                // No need to broadcast — other ranks will get their own delivery
                // from the daemon dispatcher when it processes their partials.
                // Actually for tp_size=1 there are no other ranks.
            }
        }
        // else: other ranks haven't arrived yet, daemon dispatcher will handle
        // reduction when all partials arrive and broadcast the response.
    } else {
        // Send partial to coordinator via point-to-point
        let peer_bytes = shared_state
            .peer_id_map
            .get(coordinator)
            .map(|r| r.clone());
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
            shared_state.pending_tp_partials.remove(&(request_id, layer_idx));
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

    #[test]
    fn test_allreduce_collector_single_rank() {
        let mut collector = TpAllReduceCollector::new(1);
        // Create a small f32 tensor [1.0, 2.0, 3.0], compress it
        let raw: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1).unwrap();

        let req = TpAllReduceRequest {
            request_id: Uuid::new_v4(),
            layer_idx: 0,
            tp_rank: 0,
            tp_size: 1,
            partial_data: compressed,
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
        };

        assert!(collector.insert(req, None));
        let (reduced, shape) = collector.reduce_sum().unwrap();
        assert_eq!(shape, vec![1, 1, 3]);

        // Decompress and verify
        let dec = zstd::decode_all(std::io::Cursor::new(&reduced)).unwrap();
        let vals: Vec<f32> = dec
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_allreduce_collector_two_ranks() {
        let mut collector = TpAllReduceCollector::new(2);
        let request_id = Uuid::new_v4();

        // Rank 0: [1.0, 2.0, 3.0]
        let raw0: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let c0 = zstd::encode_all(std::io::Cursor::new(&raw0), 1).unwrap();

        // Rank 1: [4.0, 5.0, 6.0]
        let raw1: Vec<u8> = [4.0f32, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let c1 = zstd::encode_all(std::io::Cursor::new(&raw1), 1).unwrap();

        let req0 = TpAllReduceRequest {
            request_id,
            layer_idx: 5,
            tp_rank: 0,
            tp_size: 2,
            partial_data: c0,
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
        };
        let req1 = TpAllReduceRequest {
            request_id,
            layer_idx: 5,
            tp_rank: 1,
            tp_size: 2,
            partial_data: c1,
            shape: vec![1, 1, 3],
            op: AllReduceOp::Sum,
        };

        assert!(!collector.insert(req0, None)); // not all arrived yet
        assert!(collector.insert(req1, None)); // all arrived

        let (reduced, shape) = collector.reduce_sum().unwrap();
        assert_eq!(shape, vec![1, 1, 3]);

        let dec = zstd::decode_all(std::io::Cursor::new(&reduced)).unwrap();
        let vals: Vec<f32> = dec
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
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

        // Delivering again should fail (already consumed)
        assert!(!registry.deliver(resp));
    }

    #[test]
    fn test_allreduce_collector_four_ranks() {
        let mut collector = TpAllReduceCollector::new(4);
        let request_id = Uuid::new_v4();

        // 4 ranks each contributing [rank, rank, rank]
        for rank in 0..4u32 {
            let val = rank as f32;
            let raw: Vec<u8> = [val, val, val]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1).unwrap();
            let req = TpAllReduceRequest {
                request_id,
                layer_idx: 0,
                tp_rank: rank,
                tp_size: 4,
                partial_data: compressed,
                shape: vec![1, 1, 3],
                op: AllReduceOp::Sum,
            };
            let all = collector.insert(req, None);
            assert_eq!(all, rank == 3); // only last one completes
        }

        let (reduced, _) = collector.reduce_sum().unwrap();
        let dec = zstd::decode_all(std::io::Cursor::new(&reduced)).unwrap();
        let vals: Vec<f32> = dec
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // 0+1+2+3 = 6.0
        assert_eq!(vals, vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_allreduce_collector_out_of_order() {
        let mut collector = TpAllReduceCollector::new(3);
        let request_id = Uuid::new_v4();

        // Insert rank 2, then 0, then 1
        for &rank in &[2u32, 0, 1] {
            let val = (rank + 1) as f32;
            let raw: Vec<u8> = [val]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1).unwrap();
            let req = TpAllReduceRequest {
                request_id,
                layer_idx: 7,
                tp_rank: rank,
                tp_size: 3,
                partial_data: compressed,
                shape: vec![1, 1, 1],
                op: AllReduceOp::Sum,
            };
            collector.insert(req, None);
        }

        let (reduced, _) = collector.reduce_sum().unwrap();
        let dec = zstd::decode_all(std::io::Cursor::new(&reduced)).unwrap();
        let vals: Vec<f32> = dec
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // 1+2+3 = 6.0
        assert_eq!(vals, vec![6.0]);
    }

    #[test]
    fn test_registry_undelivered() {
        let registry = AllReduceRegistry::new();
        let rid = Uuid::new_v4();

        // Deliver without registering — should return false
        let resp = TpAllReduceResponse {
            request_id: rid,
            layer_idx: 0,
            reduced_data: vec![],
            shape: vec![],
        };
        assert!(!registry.deliver(resp));
    }
}
