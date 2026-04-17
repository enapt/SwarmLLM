/// Collects partial AllReduce tensors from TP ranks for a single (request, layer).
/// When all `tp_size` partials arrive, the coordinator sums them and responds.
pub struct TpAllReduceCollector {
    pub tp_size: u32,
    /// Collected partials indexed by tp_rank.
    pub partials: Vec<Option<crate::types::TpAllReduceRequest>>,
    /// Sender peer bytes for responding to each rank.
    pub sender_peers: Vec<Option<Vec<u8>>>,
    pub created_at: std::time::Instant,
}

impl TpAllReduceCollector {
    pub fn new(tp_size: u32) -> Self {
        // Clamp tp_size to [1, 32] to prevent panics from empty partials vec
        // and bound memory allocation from malicious requests
        let safe_size = tp_size.clamp(1, 32) as usize;
        Self {
            tp_size,
            partials: vec![None; safe_size],
            sender_peers: vec![None; safe_size],
            created_at: std::time::Instant::now(),
        }
    }

    /// Insert a partial. Returns true when all partials have arrived.
    pub fn insert(
        &mut self,
        req: crate::types::TpAllReduceRequest,
        sender_peer: Option<Vec<u8>>,
    ) -> bool {
        let rank = req.tp_rank as usize;
        // Validate tp_rank is within bounds and tp_size matches collector's expected size
        if rank >= self.partials.len() {
            tracing::warn!(
                rank,
                tp_size = self.tp_size,
                "AllReduce: tp_rank out of bounds — ignoring"
            );
            return false;
        }
        if req.tp_size != self.tp_size {
            tracing::warn!(
                req_tp_size = req.tp_size,
                collector_tp_size = self.tp_size,
                "AllReduce: tp_size mismatch — ignoring"
            );
            return false;
        }
        if self.partials[rank].is_some() {
            tracing::warn!(rank, "AllReduce: duplicate partial for rank — overwriting");
        }
        self.sender_peers[rank] = sender_peer;
        self.partials[rank] = Some(req);
        self.partials.iter().all(|p| p.is_some())
    }

    /// Sum all partial tensors (f32) and return the reduced bytes + shape.
    pub fn reduce_sum(&self) -> Result<(Vec<u8>, Vec<u32>), crate::error::SwarmError> {
        let first = self.partials[0].as_ref().ok_or_else(|| {
            crate::error::SwarmError::Internal("AllReduce: missing rank 0 partial".into())
        })?;
        let shape = first.shape.clone();
        let elem_count: usize = shape
            .iter()
            .try_fold(1usize, |acc, &s| acc.checked_mul(s as usize))
            .ok_or_else(|| {
                crate::error::SwarmError::Internal("AllReduce: shape overflow".into())
            })?;
        // Cap at 256MB worth of f32 elements (64M floats)
        if elem_count > 64 * 1024 * 1024 {
            return Err(crate::error::SwarmError::Internal(
                "AllReduce: tensor too large".into(),
            ));
        }

        // Decompress first partial (cap decompressed size to prevent zip-bomb)
        let max_decompressed = elem_count * 4 + 1024; // expected size + small margin
        let decompressed = {
            let mut decoder = zstd::Decoder::new(std::io::Cursor::new(&first.partial_data))
                .map_err(|e| crate::error::SwarmError::Internal(format!("zstd init: {e}")))?;
            let mut buf = Vec::with_capacity(elem_count * 4);
            use std::io::Read;
            decoder
                .by_ref()
                .take(max_decompressed as u64)
                .read_to_end(&mut buf)
                .map_err(|e| crate::error::SwarmError::Internal(format!("zstd decompress: {e}")))?;
            buf
        };
        let mut sum = vec![0.0f32; elem_count];
        if decompressed.len() == elem_count * 4 {
            for (i, chunk) in decompressed.chunks_exact(4).enumerate() {
                sum[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        // Add remaining partials
        for (i, partial) in self.partials[1..].iter().enumerate() {
            let req = partial.as_ref().ok_or_else(|| {
                crate::error::SwarmError::Internal(format!(
                    "AllReduce: missing rank {} partial",
                    i + 1
                ))
            })?;
            let dec = {
                let mut decoder = zstd::Decoder::new(std::io::Cursor::new(&req.partial_data))
                    .map_err(|e| crate::error::SwarmError::Internal(format!("zstd init: {e}")))?;
                let mut buf = Vec::with_capacity(elem_count * 4);
                use std::io::Read;
                decoder
                    .by_ref()
                    .take(max_decompressed as u64)
                    .read_to_end(&mut buf)
                    .map_err(|e| {
                        crate::error::SwarmError::Internal(format!("zstd decompress: {e}"))
                    })?;
                buf
            };
            if dec.len() == elem_count * 4 {
                for (j, chunk) in dec.chunks_exact(4).enumerate() {
                    sum[j] += f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            } else {
                return Err(crate::error::SwarmError::Internal(format!(
                    "AllReduce: rank {} partial size mismatch ({} != {})",
                    i + 1,
                    dec.len(),
                    elem_count * 4
                )));
            }
        }

        // Check for NaN/Inf in reduced result (possible tensor poisoning)
        if sum.iter().any(|v| !v.is_finite()) {
            return Err(crate::error::SwarmError::Internal(
                "AllReduce result contains NaN/Inf — possible tensor poisoning".into(),
            ));
        }

        // Compress reduced result
        let raw: Vec<u8> = sum.iter().flat_map(|f| f.to_le_bytes()).collect();
        let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 1)
            .map_err(|e| crate::error::SwarmError::Internal(format!("zstd compress: {e}")))?;
        Ok((compressed, shape))
    }
}
