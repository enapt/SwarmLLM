//! Forward-pass and execution code for `SplitModel`: masks, `forward()`
//! variants, batched forward, and tensor-parallel phase execution.
//! Split out of `model.rs` so loading and execution live in separate files.

// ── Split model: loads only a range of layers from a GGUF ──

use std::collections::HashMap;

use candle_core::{IndexOp, Result as CandleResult, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::Module;

use crate::error::SwarmError;
use crate::model::lora::LoraAdapter;

use super::entry::BatchItem;
use super::kv_cache::KvCacheStore;
use super::model::SplitModel;
use super::{FfnVariant, LayerVariant, SsmState};

impl SplitModel {
    /// Build a causal mask for the given sequence length.
    ///
    /// Pre-allocates a single mask at a ceiling size (min of max_seq_len and 4096),
    /// then uses `narrow()` to slice views for smaller sequences — zero-copy.
    /// Only re-allocates if `t` exceeds the current ceiling.
    fn mask(&mut self, t: usize) -> CandleResult<Tensor> {
        // Fast path: existing mask is large enough — narrow-slice a view (no copy)
        if let Some((cached_size, ref mask)) = self.masks {
            if t <= cached_size {
                return mask.narrow(0, 0, t)?.narrow(1, 0, t);
            }
        }
        // Allocate at a reasonable ceiling to amortize future requests.
        // Cap at 4096 to avoid excessive memory (4096^2 = 16MB for u8).
        let alloc_size = t.max(self.max_seq_len.min(4096));
        let mask_data: Vec<_> = (0..alloc_size)
            .flat_map(|i| (0..alloc_size).map(move |j| u8::from(j > i)))
            .collect();
        let mask = Tensor::from_slice(&mask_data, (alloc_size, alloc_size), &self.device)?;
        self.masks = Some((alloc_size, mask.clone()));
        if t < alloc_size {
            mask.narrow(0, 0, t)?.narrow(1, 0, t)
        } else {
            Ok(mask)
        }
    }

    /// Build a causal mask with KV offset for prefix-cached inference.
    ///
    /// When the KV cache has been pre-populated with prefix tokens,
    /// query tokens (suffix) attend to all prefix positions plus earlier suffix
    /// positions with proper causal ordering.
    ///
    /// `query_len`: number of new (suffix) tokens being processed.
    /// `kv_len`: total KV length (prefix_len + query_len).
    fn mask_with_offset(&self, query_len: usize, kv_len: usize) -> CandleResult<Tensor> {
        let offset = kv_len - query_len;
        let mask: Vec<_> = (0..query_len)
            .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > offset + i)))
            .collect();
        Tensor::from_slice(&mask, (query_len, kv_len), &self.device)
    }

    /// Run the forward pass for this segment's layer range.
    ///
    /// - For the first segment: `input` is token IDs (i64 tensor, shape [1, seq_len]).
    ///   We apply the embedding lookup and return hidden states.
    /// - For intermediate segments: `input` is hidden state activations (f32, [1, seq, hidden_dim]).
    /// - For the last segment: returns logits (f32, [vocab_size]) for the last token position.
    /// - For intermediate segments: returns hidden states (f32, [1, seq, hidden_dim]).
    ///
    /// `kv_cache_store` and `request_id` provide per-request KV-cache isolation.
    /// The cache is stored externally in the `KvCacheStore`, keyed by request_id.
    pub fn forward(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        self.forward_with_lora(input, index_pos, kv_cache_store, request_id, None)
    }

    /// Speculative verify forward: run the model over γ positions and return
    /// logits at EVERY position (shape `[1, seq_len, vocab_size]`) instead of
    /// only the final one. Used by the distributed speculative-decoding path
    /// so the coordinator can accept/reject each draft without γ round trips.
    ///
    /// Used by the single-segment Item 2 path (first AND last segment on the
    /// same peer). The DSD multi-segment path (Item 12) uses
    /// `forward_verify_all_positions_pre_embedded` instead, since its input
    /// is hidden states from the previous segment.
    pub fn forward_verify_all_positions(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            false,
            true,
            None,
        )?;
        Ok(output)
    }

    /// Multi-segment DSD verify (Item 12): like `forward_verify_all_positions`
    /// but expects pre-embedded hidden state input (shape `[1, γ, hidden_dim]`)
    /// from the previous pipeline segment. Skips the embedding lookup and
    /// returns logits at every position (shape `[1, γ, vocab_size]`). Only
    /// valid on the last segment (has `output` head).
    pub fn forward_verify_all_positions_pre_embedded(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            true,
            true,
            None,
        )?;
        Ok(output)
    }

    /// SWIFT draft forward (arxiv 2410.06916): run the model with a layer
    /// skip mask. For each layer index `abs_layer` where
    /// `skip_mask[abs_layer]` is true, the hidden state passes through
    /// unchanged — no attention, no MLP, no KV-cache write. Layers outside
    /// this segment's range are ignored.
    ///
    /// Skipped layers' K/V cache entries remain whatever they were before
    /// this call. The verify pass (full forward over the same positions) is
    /// expected to re-populate them for all accepted positions.
    pub fn forward_with_skip_mask(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        skip_mask: &[bool],
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            false,
            false,
            Some(skip_mask),
        )?;
        Ok(output)
    }

    /// Forward pass with pre-embedded hidden states (local embedding privacy).
    ///
    /// The input tensor is already in hidden-state space (shape [1, seq, hidden_dim])
    /// — the embedding lookup is skipped even if this segment has `tok_embeddings`.
    /// Used when the requesting node performed embedding locally for privacy.
    pub fn forward_pre_embedded(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            None,
            None,
            true,
            false,
            None,
        )?;
        Ok(output)
    }

    /// Forward pass with optional LoRA adapter applied per-layer.
    ///
    /// When `lora_adapter` is `Some`, the adapter's low-rank deltas are applied
    /// to attention (Q/K/V/O) and MLP (gate/up/down) projections at each layer.
    pub fn forward_with_lora(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
    ) -> Result<Tensor, SwarmError> {
        let (output, _) = self.forward_inner(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            None,
        )?;
        Ok(output)
    }

    /// Inner forward pass implementation. When `capture_layers` is Some, captures
    /// hidden states at the specified absolute layer indices.
    fn forward_inner(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        self.forward_inner_impl(
            input,
            index_pos,
            kv_cache_store,
            request_id,
            lora_adapter,
            capture_layers,
            false,
            false,
            None,
        )
    }

    /// Core forward pass. When `skip_embedding` is true, the input is treated as
    /// pre-embedded hidden states even if this segment has `tok_embeddings`.
    /// When `all_positions` is true AND this is the last segment, the logits
    /// are computed at every input position and returned as `[1, seq_len,
    /// vocab]` (used by speculative-decoding verification). Default: slice to
    /// the final position only.
    /// When `skip_mask` is `Some`, layers `i` for which `skip_mask[abs_layer_i]`
    /// is true are skipped entirely (identity pass-through, no KV write) — the
    /// SWIFT self-speculative draft path.
    #[allow(clippy::too_many_arguments)]
    fn forward_inner_impl(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        lora_adapter: Option<&LoraAdapter>,
        capture_layers: Option<&std::collections::HashSet<usize>>,
        skip_embedding: bool,
        all_positions: bool,
        skip_mask: Option<&[bool]>,
    ) -> Result<(Tensor, HashMap<usize, Tensor>), SwarmError> {
        let forward_start = std::time::Instant::now();
        // Use component presence rather than layer indices for shard-aware is_first/is_last
        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Move input to model's device if needed (e.g. CPU → CUDA)
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer failed: {e}")))?;

        // Determine the hidden state to start from
        let mut layer_in = if is_first && !skip_embedding {
            // First segment: input is token IDs → apply embedding
            let mut emb = self
                .tok_embeddings
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                .forward(&input)
                .map_err(|e| SwarmError::Internal(format!("Embedding forward failed: {e}")))?;
            // Gemma models scale embeddings by sqrt(hidden_dim)
            if self.arch.use_gemma_norm() {
                let scale = (self.hidden_dim as f64).sqrt();
                emb = emb
                    .affine(scale, 0.0)
                    .map_err(|e| SwarmError::Internal(format!("Embedding scale failed: {e}")))?;
            }
            emb
        } else {
            // Non-first segment or pre-embedded: input is already hidden states
            input
        };

        // Get seq_len for mask
        let seq_len = layer_in.dim(1).map_err(SwarmError::internal)?;

        // Pre-flight check: reject sequences that exceed the model's context window
        // to avoid cryptic tensor dimension errors in attention.
        let total_seq = index_pos + seq_len;
        if total_seq > self.max_seq_len {
            return Err(SwarmError::Validation(format!(
                "Sequence length ({total_seq}) exceeds model context window ({}). \
                 Reduce your prompt or max_tokens.",
                self.max_seq_len
            )));
        }

        let num_layers = self.layers.len();
        // Build the cache key once — reused for both take and writeback (zero alloc on hot path).
        let cache_key = KvCacheStore::cache_key(&self.kv_model_key, request_id);

        // Get or create the per-request cache entry, extract the layer caches,
        // then drop the DashMap guard before running the (potentially slow) forward pass.
        // Use mem::take instead of clone to avoid copying the KV cache Vec.
        #[allow(unused_mut)]
        let (mut layer_kv_caches, mut layer_ssm_states): (
            Vec<Option<KvCache>>,
            Vec<Option<SsmState>>,
        ) = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            (
                std::mem::take(&mut entry.layers),
                std::mem::take(&mut entry.ssm_states),
            )
        };

        // Detect pre-populated prefix cache entries (KV already present from prefix
        // cache restoration). If so, build an offset causal mask that allows the
        // suffix query tokens to attend to all prefix KV positions.
        let kv_offset = layer_kv_caches
            .first()
            .and_then(|c| c.as_ref())
            .map(|c| c.current_seq_len())
            .unwrap_or(0);

        let mask = if seq_len == 1 {
            None
        } else if kv_offset > 0 {
            // Prefix cache: suffix query attends to (offset + seq_len) key positions
            Some(
                self.mask_with_offset(seq_len, kv_offset + seq_len)
                    .map_err(SwarmError::internal)?,
            )
        } else {
            Some(self.mask(seq_len).map_err(SwarmError::internal)?)
        };

        let max_seq_len = self.max_seq_len;
        let mut captured: HashMap<usize, Tensor> = HashMap::new();

        // Run through our layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let abs_layer = self.layer_start + layer_idx;
            let lora_param = lora_adapter.map(|a| (a, abs_layer));

            // SWIFT skip: identity pass-through, no attention, no MLP, no KV write.
            if let Some(mask) = skip_mask {
                if mask.get(abs_layer).copied().unwrap_or(false) {
                    continue;
                }
            }

            let layer_start_time = std::time::Instant::now();
            match layer {
                LayerVariant::Dense(lw) => {
                    let x = layer_in;
                    let residual = &x;
                    let x = lw
                        .attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;
                    let mut attn = lw
                        .forward_attn(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                            lora_param,
                        )
                        .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                    // Gemma 2 post-attention norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn = post_norm
                            .forward(&attn)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (attn + residual).map_err(SwarmError::internal)?;

                    let residual = &x;
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, lora_param)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    // Gemma 2 post-FFN norm: normalize before residual add
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
                    layer_in = (x + residual).map_err(SwarmError::internal)?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    let x = attention_norm
                        .forward(&layer_in)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;
                    let attn = attention
                        .forward_mla(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                        )
                        .map_err(|e| SwarmError::Internal(format!("mla: {e}")))?;
                    let x = (attn + &layer_in).map_err(SwarmError::internal)?;
                    let residual = &x;
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    layer_in = (ffn_out + residual).map_err(SwarmError::internal)?;
                }
                LayerVariant::Qwen35Attn {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    let x = layer_in;
                    let residual = &x;
                    let x = attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_attn_norm: {e}")))?;
                    let attn = weights
                        .forward_attn(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            &mut layer_kv_caches[layer_idx],
                            max_seq_len,
                        )
                        .map_err(|e| SwarmError::Internal(format!("q35_attn: {e}")))?;
                    let x = (attn + residual).map_err(SwarmError::internal)?;
                    let residual = &x;
                    let normed = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_post_attn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("q35_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("q35_moe: {e}")))?,
                    };
                    layer_in = (ffn_out + residual).map_err(SwarmError::internal)?;
                }
                LayerVariant::Qwen35Ssm {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    let x = layer_in;
                    let residual = &x;
                    let x = attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_ssm_norm: {e}")))?;
                    let ssm_out = weights
                        .forward_deltanet(&x, &mut layer_ssm_states[layer_idx])
                        .map_err(|e| SwarmError::Internal(format!("q35_deltanet: {e}")))?;
                    let x = (ssm_out + residual).map_err(SwarmError::internal)?;
                    let residual = &x;
                    let normed = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35_post_ssm_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("q35_ssm_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("q35_ssm_moe: {e}")))?,
                    };
                    layer_in = (ffn_out + residual).map_err(SwarmError::internal)?;
                }
            }
            tracing::trace!(
                layer = abs_layer,
                layer_ms = layer_start_time.elapsed().as_millis() as u64,
                "DIAG: layer forward complete"
            );

            // Capture hidden state if requested (zero overhead when not capturing)
            if let Some(layers_to_capture) = capture_layers {
                if layers_to_capture.contains(&abs_layer) {
                    captured.insert(abs_layer, layer_in.clone());
                }
            }
        }

        // Write the updated KV-caches and SSM states back to the store.
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
            entry.ssm_states = layer_ssm_states;
            entry.last_accessed = std::time::Instant::now();
        }

        let result = if is_last {
            // Last segment: apply final norm, extract last token, project to logits
            let norm = self
                .norm
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
            let output = self
                .output
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;

            let x = norm
                .forward(&layer_in)
                .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
            let x = if all_positions {
                x
            } else {
                x.i((.., seq_len - 1, ..))
                    .map_err(|e| SwarmError::Internal(format!("last_token_select: {e}")))?
            };
            let mut logits = output
                .forward(&x)
                .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
            // Gemma 2 final logit soft-capping: tanh(logits / cap) * cap
            if let Some(cap) = self.final_logit_softcap {
                logits = logits
                    .affine(1.0 / cap as f64, 0.0)
                    .and_then(|t| t.tanh())
                    .and_then(|t| t.affine(cap as f64, 0.0))
                    .map_err(|e| SwarmError::Internal(format!("final_logit_softcap: {e}")))?;
            }
            Ok(logits)
        } else {
            // Intermediate segment: return hidden states for next segment
            Ok(layer_in)
        };

        let forward_ms = forward_start.elapsed().as_millis() as u64;
        tracing::debug!(
            request_id,
            index_pos,
            seq_len,
            num_layers,
            is_first,
            is_last,
            kv_offset,
            forward_ms,
            "DIAG: SplitModel forward pass complete"
        );

        result.map(|t| (t, captured))
    }

    /// Tensor-parallel forward pass for a single layer.
    ///
    /// Each TP node computes only its fraction of the computation:
    /// - Attention: processes `n_head / tp_size` heads (head-parallel)
    /// - MLP: processes `intermediate_dim / tp_size` columns (column-parallel gate/up,
    ///   row-parallel down)
    ///
    /// Returns a **partial** hidden state that must be summed (AllReduced) across all
    /// Forward pass for multimodal (vision + text) inference.
    ///
    /// If this is the first segment and `vision_embeddings` is provided, the input
    /// is expected to contain TWO segments of token IDs separated by a marker:
    /// `[before_image_tokens...][MARKER=-1][after_image_tokens...]`
    ///
    /// The marker value -1 (as i64) indicates where vision embeddings should be
    /// inserted. Each part is embedded separately, then concatenated:
    /// `[before_emb][vision_emb][after_emb]`
    ///
    /// This matches llama.cpp's LLaVA approach of splitting the prompt at `<image>`.
    ///
    /// `vision_embeddings` shape: (num_image_tokens, hidden_dim)
    pub fn forward_multimodal(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        vision_embeddings: Option<&Tensor>,
    ) -> Result<Tensor, SwarmError> {
        let is_first = self.tok_embeddings.is_some();

        match (is_first, vision_embeddings) {
            (true, Some(vision_emb)) => {
                let input_dev = input
                    .to_device(&self.device)
                    .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

                // Read the token IDs to find the -1 marker
                let token_ids: Vec<i64> = input_dev
                    .flatten_all()
                    .map_err(|e| SwarmError::Internal(format!("flatten: {e}")))?
                    .to_vec1()
                    .map_err(|e| SwarmError::Internal(format!("to_vec1: {e}")))?;

                let marker_pos = token_ids.iter().position(|&id| id == -1);

                let tok_emb_layer = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?;

                let merged = if let Some(pos) = marker_pos {
                    // Split at marker: embed before and after separately, insert vision between
                    let num_vision = vision_emb
                        .dim(0)
                        .map_err(|e| SwarmError::Internal(format!("vision dim: {e}")))?;
                    let hidden = vision_emb
                        .dim(1)
                        .map_err(|e| SwarmError::Internal(format!("vision dim1: {e}")))?;

                    // All parts must be (1, seq, hidden) for cat along dim 1
                    let vision_3d = vision_emb
                        .reshape(&[1, num_vision, hidden])
                        .map_err(|e| SwarmError::Internal(format!("vision reshape: {e}")))?;

                    let mut parts: Vec<Tensor> = Vec::new();

                    // Embed tokens before <image>
                    if pos > 0 {
                        let before_ids = Tensor::new(&token_ids[..pos], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("before tensor: {e}")))?
                            .reshape(&[1, pos])
                            .map_err(|e| SwarmError::Internal(format!("before reshape: {e}")))?;
                        let mut before_emb = tok_emb_layer
                            .forward(&before_ids)
                            .map_err(|e| SwarmError::Internal(format!("before embed: {e}")))?;
                        // Apply Gemma embedding scale (must match forward_inner_impl)
                        if self.arch.use_gemma_norm() {
                            let scale = (self.hidden_dim as f64).sqrt();
                            before_emb = before_emb
                                .affine(scale, 0.0)
                                .map_err(|e| SwarmError::Internal(format!("gemma scale: {e}")))?;
                        }
                        // Ensure 3D: (1, pos, hidden)
                        let before_3d = if before_emb.dims().len() == 2 {
                            before_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("before unsqueeze: {e}"))
                            })?
                        } else {
                            before_emb
                        };
                        parts.push(before_3d);
                    }

                    // Insert vision embeddings
                    parts.push(vision_3d);

                    // Embed tokens after <image>
                    let after_len = token_ids.len() - pos - 1;
                    if after_len > 0 {
                        let after_ids = Tensor::new(&token_ids[pos + 1..], &self.device)
                            .map_err(|e| SwarmError::Internal(format!("after tensor: {e}")))?
                            .reshape(&[1, after_len])
                            .map_err(|e| SwarmError::Internal(format!("after reshape: {e}")))?;
                        let mut after_emb = tok_emb_layer
                            .forward(&after_ids)
                            .map_err(|e| SwarmError::Internal(format!("after embed: {e}")))?;
                        // Apply Gemma embedding scale (must match forward_inner_impl)
                        if self.arch.use_gemma_norm() {
                            let scale = (self.hidden_dim as f64).sqrt();
                            after_emb = after_emb
                                .affine(scale, 0.0)
                                .map_err(|e| SwarmError::Internal(format!("gemma scale: {e}")))?;
                        }
                        let after_3d = if after_emb.dims().len() == 2 {
                            after_emb.unsqueeze(0).map_err(|e| {
                                SwarmError::Internal(format!("after unsqueeze: {e}"))
                            })?
                        } else {
                            after_emb
                        };
                        parts.push(after_3d);
                    }

                    let refs: Vec<&Tensor> = parts.iter().collect();
                    tracing::info!(
                        request_id,
                        marker_pos = pos,
                        before_tokens = pos,
                        after_tokens = after_len,
                        vision_tokens = num_vision,
                        "VLM: inserting vision embeddings at <image> position"
                    );
                    Tensor::cat(&refs, 1)
                        .map_err(|e| SwarmError::Internal(format!("merge cat: {e}")))?
                } else {
                    // No marker found — fallback to prepending vision before text
                    let text_embeddings = tok_emb_layer
                        .forward(&input_dev)
                        .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;
                    crate::inference::vision::merge_vision_text_embeddings(
                        &text_embeddings,
                        vision_emb,
                        &[],
                    )?
                };

                tracing::info!(
                    request_id,
                    merged_seq_len = ?merged.dims(),
                    "VLM: merged embeddings ready for transformer"
                );

                // Run through layers with merged hidden states.
                // Temporarily remove tok_embeddings so forward() treats input as hidden states.
                let tok_emb = self.tok_embeddings.take();
                let result = self.forward(&merged, index_pos, kv_cache_store, request_id);
                self.tok_embeddings = tok_emb;
                result
            }
            _ => {
                // No vision embeddings or not the first segment — regular forward
                self.forward(input, index_pos, kv_cache_store, request_id)
            }
        }
    }

    /// Run a batched forward pass for multiple requests.
    ///
    /// Supports two homogeneous batch shapes:
    /// - **Decode batch**: every item has `seq_len = 1` (no mask needed).
    /// - **Prefill-chunk batch (Item 7 Phase 4)**: every item has the same
    ///   `seq_len > 1` AND the same `index_pos`. One causal mask (built once
    ///   from the first slot's KV length) serves every per-request attention
    ///   call because same index_pos ⇒ same kv_offset ⇒ same mask.
    ///
    /// Stacks inputs along the batch dimension so that MLP/norm computations
    /// benefit from GPU parallelism. Attention is still per-request because
    /// each request has its own KV-cache.
    ///
    /// Returns one output tensor per request in the same order as `items`.
    /// Falls back to sequential `forward()` when the batch is heterogeneous
    /// (mixed seq_lens or differing index_pos for seq_len > 1).
    pub fn forward_batch(
        &mut self,
        items: &[BatchItem<'_>],
        kv_cache_store: &KvCacheStore,
    ) -> Result<Vec<Tensor>, SwarmError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        // Single-item fast path: no stacking benefit.
        if items.len() == 1 {
            let item = &items[0];
            let out = self.forward(item.input, item.index_pos, kv_cache_store, item.request_id)?;
            return Ok(vec![out]);
        }

        let is_first = self.tok_embeddings.is_some();
        let is_last = self.output.is_some();

        // Convert each input to its hidden state (apply embedding if first segment)
        let mut per_request: Vec<Tensor> = Vec::with_capacity(items.len());
        for item in items {
            let input = item
                .input
                .to_device(&self.device)
                .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;
            let hidden = if is_first {
                let mut emb = self
                    .tok_embeddings
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing embedding table".into()))?
                    .forward(&input)
                    .map_err(|e| SwarmError::Internal(format!("Embedding: {e}")))?;
                // Apply Gemma embedding scale (sqrt(hidden_dim)) — matches forward_inner_impl
                if self.arch.use_gemma_norm() {
                    let scale = (self.hidden_dim as f64).sqrt();
                    emb = emb
                        .affine(scale, 0.0)
                        .map_err(|e| SwarmError::Internal(format!("Gemma scale: {e}")))?;
                }
                emb
            } else {
                input
            };
            per_request.push(hidden);
        }

        // Homogeneity check: batching only kicks in when every item shares
        // (seq_len, index_pos). Mixed batches fall back to sequential forwards
        // so a slow slot doesn't block the fast ones.
        let first_seq_len = per_request[0].dim(1).unwrap_or(0);
        let first_index_pos = items[0].index_pos;
        let all_same_seq = per_request
            .iter()
            .all(|t| t.dim(1).unwrap_or(0) == first_seq_len);
        let all_same_pos = items.iter().all(|i| i.index_pos == first_index_pos);
        let homogeneous = first_seq_len > 0 && all_same_seq && all_same_pos;

        if !homogeneous {
            // Mixed seq_lens or differing index_pos: fall back to sequential.
            let mut results = Vec::with_capacity(items.len());
            for item in items {
                results.push(self.forward(
                    item.input,
                    item.index_pos,
                    kv_cache_store,
                    item.request_id,
                )?);
            }
            return Ok(results);
        }

        let seq_len = first_seq_len;

        // Context window check: reject any item whose index_pos + seq_len
        // exceeds max_seq_len (same guard as forward_inner_impl, prevents RoPE
        // table out-of-bounds).
        for item in items {
            if item.index_pos + seq_len > self.max_seq_len {
                return Err(SwarmError::Validation(format!(
                    "Batch item index_pos+seq_len ({}) exceeds model context window ({})",
                    item.index_pos + seq_len,
                    self.max_seq_len
                )));
            }
        }

        let batch_size = items.len();
        // Clone rather than borrow so the later `self.mask(...)` mutable borrow
        // doesn't conflict with reuse of this key below.
        let model_key: String = self.kv_model_key.clone();
        let num_layers = self.layers.len();

        // Extract all per-request KV-caches and SSM states up front (drop DashMap guards immediately).
        // Use mem::take instead of clone to avoid deep-copying all KV tensors.
        let mut all_kv_caches: Vec<Vec<Option<KvCache>>> = Vec::with_capacity(batch_size);
        let mut all_ssm_states: Vec<Vec<Option<SsmState>>> = Vec::with_capacity(batch_size);
        for item in items.iter() {
            let key = KvCacheStore::cache_key(&model_key, item.request_id);
            let mut entry = kv_cache_store.get_or_create_keyed(&key, num_layers);
            entry.last_accessed = std::time::Instant::now();
            all_kv_caches.push(std::mem::take(&mut entry.layers));
            all_ssm_states.push(std::mem::take(&mut entry.ssm_states));
        }

        let max_seq_len = self.max_seq_len;

        // Build one causal mask for the whole batch when seq_len > 1. Every
        // slot shares (seq_len, index_pos) at this point, so they also share
        // `kv_offset` (== first slot's layer-0 KV length), hence share a mask.
        let mask = if seq_len == 1 {
            None
        } else {
            let kv_offset = all_kv_caches
                .first()
                .and_then(|per_layer| per_layer.first())
                .and_then(|l| l.as_ref())
                .map(|c| c.current_seq_len())
                .unwrap_or(0);
            if kv_offset > 0 {
                Some(
                    self.mask_with_offset(seq_len, kv_offset + seq_len)
                        .map_err(SwarmError::internal)?,
                )
            } else {
                Some(self.mask(seq_len).map_err(SwarmError::internal)?)
            }
        };

        // Stack all hidden states into a single batch tensor:
        // [batch, seq_len, hidden_dim] (seq_len is 1 for decode, >1 for prefill chunks).
        let batch_refs: Vec<&Tensor> = per_request.iter().collect();
        let mut batched = Tensor::cat(&batch_refs, 0)
            .map_err(|e| SwarmError::Internal(format!("Batch stack: {e}")))?;

        // Process through layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match layer {
                LayerVariant::Dense(lw) => {
                    let residual = batched.clone();
                    let normed = lw
                        .attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = lw
                            .forward_attn(
                                &x_i,
                                mask.as_ref(),
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                                None,
                            )
                            .map_err(|e| SwarmError::Internal(format!("attn: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let mut attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("attn restack: {e}")))?;
                    if let Some(ref post_norm) = lw.post_attention_norm {
                        attn_batched = post_norm
                            .forward(&attn_batched)
                            .map_err(|e| SwarmError::Internal(format!("post_attn_norm: {e}")))?;
                    }
                    let x = (&attn_batched + &residual).map_err(SwarmError::internal)?;

                    let residual2 = x.clone();
                    let x = lw
                        .ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ffn_norm: {e}")))?;
                    let mut x = match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("moe: {e}")))?,
                    };
                    if let Some(ref post_norm) = lw.post_ffw_norm {
                        x = post_norm
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("post_ffw_norm: {e}")))?;
                    }
                    batched = (&x + &residual2).map_err(SwarmError::internal)?;
                }
                LayerVariant::DeepSeek {
                    attention,
                    ffn,
                    attention_norm,
                    ffn_norm,
                } => {
                    // DeepSeek batch: per-request attention (MLA), batched FFN
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("ds_attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = attention
                            .forward_mla(
                                &x_i,
                                mask.as_ref(),
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                            )
                            .map_err(|e| SwarmError::Internal(format!("mla_batch: {e}")))?;
                        attn_outputs.push(attn_out);
                    }

                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("mla restack: {e}")))?;
                    let x = (&attn_batched + &batched).map_err(SwarmError::internal)?;

                    let residual = x.clone();
                    let normed = ffn_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("ds_ffn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed, None)
                            .map_err(|e| SwarmError::Internal(format!("ds_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed)
                            .map_err(|e| SwarmError::Internal(format!("moe_batch: {e}")))?,
                    };
                    batched = (&ffn_out + &residual).map_err(SwarmError::internal)?;
                }
                LayerVariant::Qwen35Attn {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    // Qwen 3.5 attention: per-request attention + batched FFN (same as Dense pattern)
                    let residual = batched.clone();
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("q35b_attn_norm: {e}")))?;

                    let mut attn_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, item) in items.iter().enumerate() {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let attn_out = weights
                            .forward_attn(
                                &x_i,
                                mask.as_ref(),
                                item.index_pos,
                                &mut all_kv_caches[req_idx][layer_idx],
                                max_seq_len,
                            )
                            .map_err(|e| SwarmError::Internal(format!("q35b_attn: {e}")))?;
                        attn_outputs.push(attn_out);
                    }
                    let attn_refs: Vec<&Tensor> = attn_outputs.iter().collect();
                    let attn_batched = Tensor::cat(&attn_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("q35b_attn_restack: {e}")))?;
                    let x = (&attn_batched + &residual).map_err(SwarmError::internal)?;

                    let residual2 = x.clone();
                    let normed2 = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35b_post_attn_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed2, None)
                            .map_err(|e| SwarmError::Internal(format!("q35b_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed2)
                            .map_err(|e| SwarmError::Internal(format!("q35b_moe: {e}")))?,
                    };
                    batched = (&ffn_out + &residual2).map_err(SwarmError::internal)?;
                }
                LayerVariant::Qwen35Ssm {
                    ref weights,
                    ref ffn,
                    ref attention_norm,
                    ref post_attention_norm,
                } => {
                    // Qwen 3.5 SSM: per-request DeltaNet (SSM state is per-request) + batched FFN
                    let residual = batched.clone();
                    let normed = attention_norm
                        .forward(&batched)
                        .map_err(|e| SwarmError::Internal(format!("q35b_ssm_norm: {e}")))?;

                    let mut ssm_outputs: Vec<Tensor> = Vec::with_capacity(batch_size);
                    for (req_idx, req_ssm) in all_ssm_states.iter_mut().enumerate().take(batch_size)
                    {
                        let x_i = normed
                            .narrow(0, req_idx, 1)
                            .map_err(|e| SwarmError::Internal(format!("narrow: {e}")))?;
                        let ssm_out = weights
                            .forward_deltanet(&x_i, &mut req_ssm[layer_idx])
                            .map_err(|e| SwarmError::Internal(format!("q35b_deltanet: {e}")))?;
                        ssm_outputs.push(ssm_out);
                    }
                    let ssm_refs: Vec<&Tensor> = ssm_outputs.iter().collect();
                    let ssm_batched = Tensor::cat(&ssm_refs, 0)
                        .map_err(|e| SwarmError::Internal(format!("q35b_ssm_restack: {e}")))?;
                    let x = (&ssm_batched + &residual).map_err(SwarmError::internal)?;

                    let residual2 = x.clone();
                    let normed2 = post_attention_norm
                        .forward(&x)
                        .map_err(|e| SwarmError::Internal(format!("q35b_post_ssm_norm: {e}")))?;
                    let ffn_out = match ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&normed2, None)
                            .map_err(|e| SwarmError::Internal(format!("q35b_ssm_mlp: {e}")))?,
                        FfnVariant::MoE(moe) => moe
                            .forward(&normed2)
                            .map_err(|e| SwarmError::Internal(format!("q35b_ssm_moe: {e}")))?,
                    };
                    batched = (&ffn_out + &residual2).map_err(SwarmError::internal)?;
                }
            }
        }

        // Write updated KV-caches and SSM states back (take instead of clone to avoid copying)
        for (req_idx, item) in items.iter().enumerate() {
            let key = KvCacheStore::cache_key(&model_key, item.request_id);
            let mut entry = kv_cache_store.get_or_create_keyed(&key, num_layers);
            entry.layers = std::mem::take(&mut all_kv_caches[req_idx]);
            entry.ssm_states = std::mem::take(&mut all_ssm_states[req_idx]);
            entry.last_accessed = std::time::Instant::now();
        }

        // Split batch back into per-request outputs
        let mut results = Vec::with_capacity(batch_size);
        for req_idx in 0..batch_size {
            let per_req = batched
                .narrow(0, req_idx, 1)
                .map_err(|e| SwarmError::Internal(format!("split: {e}")))?;

            if is_last {
                let norm = self
                    .norm
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing final norm".into()))?;
                let output = self
                    .output
                    .as_ref()
                    .ok_or_else(|| SwarmError::Internal("Missing output head".into()))?;
                let x = norm
                    .forward(&per_req)
                    .map_err(|e| SwarmError::Internal(format!("final_norm: {e}")))?;
                // Slice the LAST position of this request (decode: seq_len=1 ⇒ 0;
                // prefill chunk: seq_len>1 ⇒ seq_len-1). Matches
                // `forward_inner_impl`'s non-`all_positions` output path.
                let x = x
                    .i((.., seq_len - 1, ..))
                    .map_err(|e| SwarmError::Internal(format!("last_token: {e}")))?;
                let mut logits = output
                    .forward(&x)
                    .map_err(|e| SwarmError::Internal(format!("output_proj: {e}")))?;
                // Apply final logit softcap for Gemma 2 (must match forward_inner_impl)
                if let Some(cap) = self.final_logit_softcap {
                    logits = logits
                        .affine(1.0 / cap as f64, 0.0)
                        .and_then(|t| t.tanh())
                        .and_then(|t| t.affine(cap as f64, 0.0))
                        .map_err(|e| SwarmError::Internal(format!("final_logit_softcap: {e}")))?;
                }
                results.push(logits);
            } else {
                results.push(per_req);
            }
        }

        Ok(results)
    }

    /// Return a reference to the loaded vocabulary, if available.
    /// Pre-split all layer weights for tensor parallelism.
    ///
    /// Reduces VRAM per rank by splitting attention heads and FFN columns.
    /// Must be called once before any TP forward passes.
    pub fn pre_split_for_tp(&mut self, tp_rank: usize, tp_size: usize) -> Result<(), SwarmError> {
        if tp_size <= 1 {
            return Ok(());
        }
        let device = self.device.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if let LayerVariant::Dense(ref mut lw) = layer {
                lw.pre_split_for_tp(tp_rank, tp_size, &device)
                    .map_err(|e| SwarmError::Internal(format!("TP split layer {i}: {e}")))?;
            }
        }
        tracing::info!(
            tp_rank,
            tp_size,
            layers = self.layers.len(),
            "Pre-split weights for tensor parallelism"
        );
        Ok(())
    }

    /// Forward a single layer in a specific TP phase (AttnOnly or FfnOnly).
    ///
    /// Unlike the full `forward()`, this processes exactly ONE layer and does NOT
    /// add the residual connection — the coordinator adds residuals after AllReduce.
    ///
    /// - **AttnOnly**: attention_norm → head-sliced attention → return partial
    /// - **FfnOnly**: ffn_norm → column-sliced FFN → return partial
    ///
    /// The model must have been pre-split via `pre_split_for_tp()`.
    pub fn forward_tp_phase(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        kv_cache_store: &KvCacheStore,
        request_id: &str,
        abs_layer: usize,
        phase: &crate::types::TpPhase,
    ) -> Result<Tensor, SwarmError> {
        let input = input
            .to_device(&self.device)
            .map_err(|e| SwarmError::Internal(format!("Device transfer: {e}")))?;

        // EmbedOnly: just tokenize + embed, no layer processing
        if *phase == crate::types::TpPhase::EmbedOnly {
            if let Some(ref emb) = self.tok_embeddings {
                let mut result = emb
                    .forward(&input)
                    .map_err(|e| SwarmError::Internal(format!("tp embed: {e}")))?;
                if self.arch.use_gemma_norm() {
                    let scale = (self.hidden_dim as f64).sqrt();
                    result = result
                        .affine(scale, 0.0)
                        .map_err(|e| SwarmError::Internal(format!("tp embed scale: {e}")))?;
                }
                return Ok(result);
            } else {
                return Err(SwarmError::Internal(
                    "EmbedOnly requested but model has no tok_embeddings".into(),
                ));
            }
        }

        let local_layer_idx = abs_layer.checked_sub(self.layer_start).ok_or_else(|| {
            SwarmError::Internal(format!(
                "TP layer {abs_layer} below model range [{}, {})",
                self.layer_start,
                self.layer_start + self.layers.len()
            ))
        })?;
        if local_layer_idx >= self.layers.len() {
            return Err(SwarmError::Internal(format!(
                "TP layer {abs_layer} above model range [{}, {})",
                self.layer_start,
                self.layer_start + self.layers.len()
            )));
        }

        let seq_len = input.dim(1).map_err(SwarmError::internal)?;

        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len).map_err(SwarmError::internal)?)
        };

        let num_layers = self.layers.len();
        let cache_key = KvCacheStore::cache_key(&self.kv_model_key, request_id);
        let mut layer_kv_caches = {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            std::mem::take(&mut entry.layers)
        };

        let result = match &self.layers[local_layer_idx] {
            LayerVariant::Dense(lw) => match phase {
                crate::types::TpPhase::AttnOnly => {
                    let x = lw
                        .attention_norm
                        .forward(&input)
                        .map_err(|e| SwarmError::Internal(format!("tp attn_norm: {e}")))?;
                    lw.forward_attn(
                        &x,
                        mask.as_ref(),
                        index_pos,
                        &mut layer_kv_caches[local_layer_idx],
                        self.max_seq_len,
                        None,
                    )
                    .map_err(|e| SwarmError::Internal(format!("tp attn: {e}")))
                }
                crate::types::TpPhase::FfnOnly => {
                    let x = lw
                        .ffn_norm
                        .forward(&input)
                        .map_err(|e| SwarmError::Internal(format!("tp ffn_norm: {e}")))?;
                    match &lw.ffn {
                        FfnVariant::Dense(mlp) => mlp
                            .forward(&x, None)
                            .map_err(|e| SwarmError::Internal(format!("tp mlp: {e}"))),
                        FfnVariant::MoE(moe) => moe
                            .forward(&x)
                            .map_err(|e| SwarmError::Internal(format!("tp moe: {e}"))),
                    }
                }
                crate::types::TpPhase::Full | crate::types::TpPhase::EmbedOnly => {
                    Err(SwarmError::Internal(
                        "TpPhase::Full/EmbedOnly not valid for single-layer TP".into(),
                    ))
                }
            },
            _ => Err(SwarmError::Internal(
                "TP not supported for non-Dense layers".into(),
            )),
        }?;

        // Write back KV caches
        {
            let mut entry = kv_cache_store.get_or_create_keyed(&cache_key, num_layers);
            entry.layers = layer_kv_caches;
        }

        Ok(result)
    }
}
