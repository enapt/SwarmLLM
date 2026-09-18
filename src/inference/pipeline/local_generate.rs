//! The plan came back to this node: run it the way a local request runs.
//!
//! A node holding every shard of a model still goes through the router
//! whenever it would run that model on its PROCESSOR and has peers to ask —
//! [`SharedState::local_fast_path_for`] stands the API fast path aside so the
//! scheduler gets to consider delegating. That is deliberate and right. Its
//! doc says the cost when nobody better is found is "only a scheduling pass".
//!
//! It was not. The plan came back naming this node for all of it, and the
//! pipeline then ran it as a single segment: a `LayerForward` per token into
//! our own worker, through the dispatcher, for the whole reply. Which loses
//! everything the `Generate` path does and the forward path does not —
//! **the prefix cache above all**, so a CPU node with peers recomputed the
//! full prompt on every turn of every conversation, for ever. Report #018
//! measured the consequence: `prefix-cache HIT` appears zero times in a week
//! of logs across several models, on a node holding a complete 14B.
//! Continuous batching, slot admission and n-gram speculation live on the same
//! path and were lost with it.
//!
//! So a single-segment plan naming this node is executed as the local generate
//! it is. Nothing about the scheduler's decision changes: it still runs, still
//! prices the swarm, and still delegates whenever a peer is worth it. This is
//! only about how its answer is carried out when the answer is "here".
//!
//! `remote_generate::eligible` has asserted this arrangement since it was
//! written — it excludes the local node with "local inference is handled by
//! `execute_local`, which has its own faster path". That was true only of
//! requests that never reached the router.

use crate::error::SwarmError;
use crate::inference::router::InferenceOutput;
use crate::inference::router::StreamingTokenTx;

use super::PipelineExecutor;

impl PipelineExecutor {
    /// Run the whole request on this node, as one generation, when the plan is
    /// a single segment naming this node and covering the complete model.
    ///
    /// `Ok(None)` means not eligible — the caller falls through to the ordinary
    /// per-token pipeline, which stays correct for every shape this declines.
    pub(super) async fn try_local_generate_fastpath(
        &mut self,
        token_tx: Option<StreamingTokenTx>,
    ) -> Result<Option<InferenceOutput>, SwarmError> {
        let Some(layer_range) = self.local_whole_model_segment() else {
            return Ok(None);
        };

        let request_id = self.request.id;
        let model_id = self.request.model_id.clone();
        // The prompt AND the stop strings its template implies, together. The
        // pairing is the point: a marker we do not pass is one nothing will
        // match, and the local worker truncates only the stop list it is
        // handed.
        let (prompt, sampling) = self.build_prompt_and_stops().await;

        tracing::info!(
            %request_id,
            model = %model_id,
            layer_start = layer_range.0,
            layer_end = layer_range.1,
            "DIAG: single-segment plan names this node — running it as a local generate"
        );

        // **Watched, not merely awaited.** This is a wait that runs for as long
        // as the whole reply takes — minutes on a processor — and
        // `generate_attempt` loops on the worker's messages until the worker
        // stops. It does not read the cancel flag and a closed token channel
        // does not end it (`let _ = tx.send(..)`), so nothing here would notice
        // a client that left: the API fast path drops the same future through a
        // `select!` on its disconnect watch, and this path had no equivalent.
        //
        // Dropping it is the mechanism, and it works because the pool's
        // `ResponseGuard` sends `CancelRequest` to the worker on drop.
        //
        // `session_id` is carried through so a multi-turn conversation keeps
        // its KV entry, exactly as it does on the API fast path.
        let pool = &self.shared_state.model_process_pool;
        let out = crate::inference::cancel::unless_cancelled(
            pool.generate(
                &model_id,
                layer_range,
                prompt,
                sampling,
                request_id,
                self.request.session_id.clone(),
                token_tx,
            ),
            self.request.cancel.as_ref(),
        )
        .await?;
        Ok(Some(out))
    }

    /// The layer range to generate over, when the plan is one segment, on this
    /// node, spanning the complete local split model. `None` otherwise.
    ///
    /// The span must be checked rather than assumed. A segment that does not
    /// reach both ends produces hidden states, not tokens, and handing it to
    /// `generate` would silently answer with the output of a fraction of the
    /// model.
    fn local_whole_model_segment(&self) -> Option<(u32, u32)> {
        // Tensor-parallel groups, a LoRA adapter and images each need the
        // pipeline's own machinery; the remote fast path declines them for the
        // same reason.
        if super::fastpath_request_disqualified(self) {
            return None;
        }
        if self.assignment.segments.len() != 1 {
            return None;
        }
        let segment = &self.assignment.segments[0];
        if segment.node_id != *self.shared_state.identity.node_id() {
            return None;
        }
        // Prompt privacy is NOT a disqualifier here, and that is not an
        // oversight. What it guarantees is that the prompt and the sampled
        // tokens stay on this machine; a plan that runs every layer here
        // satisfies it completely, with no boomerang to build. The remote
        // sibling declines on it because that path puts the raw prompt on the
        // wire.
        // A registered entry answers directly, and is preferred: `is_complete`
        // is the strongest form of the question, and the lookup carries the
        // predicate so the decision and the range acted on cannot disagree
        // (gotcha #187).
        if let Some(meta) =
            crate::api::openai::get_split_model_meta(&self.shared_state, &self.request.model_id)
        {
            return (meta.layer_range == segment.layer_range).then_some(meta.layer_range);
        }

        // **The absence of a registration does not mean this node cannot run
        // the model.** `split_models` has one writer, `auto_manage::scan`, and
        // it refuses to register past a budget — a ceiling on what the node
        // OFFERS, sized as though every registered model were resident at once
        // even though the entry allocates nothing. On a node holding several
        // models the later ones therefore never get an entry, and reading that
        // absence as "not ours" left this fast path firing ZERO times in a full
        // day's log while the scheduler kept assigning this node single local
        // segments (report #018, reopened 2026-09-17). What is lost is the
        // prefix cache, continuous batching, slot admission and n-gram
        // speculation — exactly what #018 was filed about.
        //
        // A ceiling on what to OFFER must not decide how a request already
        // assigned here is EXECUTED. The pipeline is going to run this very
        // segment on this very node either way, so this adds no failure mode
        // the other branch does not already have — only the SPAN has to be
        // right, and a segment short of either end produces hidden states
        // rather than tokens.
        //
        // Everything below comes from the MANIFEST, which is also where
        // `scan.rs` derives `is_first`/`is_last`, so the fallback cannot
        // disagree with the registration it stands in for. Holding shard 0 and
        // the last shard is what makes the worker's embedding table and output
        // head present; without them a whole-model range would load a model
        // with no `tok_embeddings` and push token ids into the first block,
        // which is gotcha #187's crash.
        let registry = &self.shared_state.model_registry;
        let manifest = registry.get_manifest(&self.request.model_id)?;
        let held = registry.local_shard_indices_in(&manifest, self.shared_state.identity.node_id());
        let last_shard = manifest.shard_count.saturating_sub(1);
        if !held.contains(&0) || !held.contains(&last_shard) {
            return None;
        }
        let total_layers = manifest.num_layers;
        (segment.layer_range == (0, total_layers)).then_some(segment.layer_range)
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::pipeline::PipelineExecutor;
    use crate::inference::split::SplitModelEntry;
    use crate::types::*;
    use std::sync::Arc;

    fn complete_entry(layer_start: usize, layer_end: usize) -> SplitModelEntry {
        SplitModelEntry {
            last_used: std::sync::atomic::AtomicU64::new(0),
            estimated_vram_mb: 0,
            is_complete: true,
            eos_tokens: vec![],
            eos_token_str: String::new(),
            bos_token: String::new(),
            cached_chat_template: None,
            vocab: None,
            layer_start,
            layer_end,
        }
    }

    fn executor_for(
        state: Arc<crate::daemon::SharedState>,
        segments: Vec<PipelineSegment>,
    ) -> PipelineExecutor {
        let request = InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model_id: ModelId("m".into()),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".into(),
                images: vec![],
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: state.identity.node_id().clone(),
            priority: PriorityTier::Silver,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
            tools: None,
            cancel: None,
            route_override: None,
        };
        let (tx, _rx) = tokio::sync::mpsc::channel::<NetworkCommand>(8);
        let assignment = PipelineAssignment {
            request_id: request.id,
            segments,
            standbys: vec![],
            tp_groups: vec![],
            supports_speculative: false,
        };
        PipelineExecutor::new(state, tx, request, assignment)
    }

    fn segment(node_id: NodeId, layer_range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id,
            shard_id: ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            layer_range,
        }
    }

    /// A plan that names this node for the whole model is a local generate.
    /// Before this, it was a `LayerForward` per token into our own worker —
    /// the one path that never consults the prefix cache (report #018).
    #[tokio::test]
    async fn a_single_local_segment_spanning_the_model_is_a_local_generate() {
        let state = crate::inference::pipeline::tests::make_test_state();
        state
            .split_models
            .insert((ModelId("m".into()), 0, 48), complete_entry(0, 48));
        let local = state.identity.node_id().clone();
        let exec = executor_for(state, vec![segment(local, (0, 48))]);
        assert_eq!(exec.local_whole_model_segment(), Some((0, 48)));
    }

    /// Build a manifest for `m` and record this node as holding every shard.
    fn register_whole_model(
        state: &Arc<crate::daemon::SharedState>,
        num_layers: u32,
        shard_count: u32,
    ) {
        let model_id = ModelId("m".into());
        let shards: Vec<ShardInfo> = (0..shard_count)
            .map(|index| ShardInfo {
                index,
                layer_range: (0, num_layers),
                size_bytes: 1,
                hash: [0u8; 32],
                tensors: vec![],
            })
            .collect();
        state.model_registry.register_manifest(ModelManifest {
            id: model_id.clone(),
            name: "Test Model".into(),
            architecture: ModelArchitecture::Llama,
            num_layers,
            num_params_billions: 1.0,
            quantization: Quantization::Q4KM,
            total_size_bytes: 1,
            shard_count,
            shards,
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
        });
        for index in 0..shard_count {
            state.model_registry.record_shard_holder(
                ShardId {
                    model_id: model_id.clone(),
                    index,
                },
                state.identity.node_id().clone(),
            );
        }
    }

    /// **A model with no `split_models` entry is still ours to run.**
    ///
    /// `split_models` is written only by `auto_manage::scan`, which refuses to
    /// register past a budget sized as though every registered model were
    /// resident at once. So on a node holding several models the later ones
    /// have no entry at all — and requiring one left this fast path firing zero
    /// times in a full day's log on a node the scheduler kept assigning single
    /// local segments (report #018, reopened 2026-09-17). Without the fallback
    /// this returns `None` and the reply is computed a `LayerForward` at a
    /// time, with no prefix cache, no batching and no speculation.
    #[tokio::test]
    async fn an_unregistered_whole_model_is_still_a_local_generate() {
        let state = crate::inference::pipeline::tests::make_test_state();
        register_whole_model(&state, 48, 2);
        assert!(
            state.split_models.is_empty(),
            "the point of this test is the ABSENCE of a registration"
        );
        let local = state.identity.node_id().clone();
        let exec = executor_for(state, vec![segment(local, (0, 48))]);
        assert_eq!(exec.local_whole_model_segment(), Some((0, 48)));
    }

    /// The fallback must not hand `generate` a model whose ends are missing.
    ///
    /// Shard 0 carries `token_embd.weight` and the last shard the output head.
    /// A worker loading a whole-model range without them has no
    /// `tok_embeddings`, passes the input through unchanged, and pushes raw
    /// token ids into the first block's rms-norm — gotcha #187's crash
    /// (`shape mismatch in rms-norm [1, 128] [3072]`). Declining here is free:
    /// the pipeline simply runs the segment the way it did before.
    #[tokio::test]
    async fn the_fallback_declines_when_an_end_shard_is_missing() {
        let state = crate::inference::pipeline::tests::make_test_state();
        register_whole_model(&state, 48, 3);
        // Drop the LAST shard's holder record: the output head is not here.
        state.model_registry.remove_shard_holder(
            &ShardId {
                model_id: ModelId("m".into()),
                index: 2,
            },
            state.identity.node_id(),
        );
        let local = state.identity.node_id().clone();
        let exec = executor_for(state, vec![segment(local, (0, 48))]);
        assert_eq!(exec.local_whole_model_segment(), None);
    }

    /// A segment that does not reach both ends produces hidden states, not
    /// tokens. Handing it to `generate` would answer with the output of a
    /// fraction of the model, which is why the span is checked and not assumed
    /// from the segment count.
    #[tokio::test]
    async fn a_partial_local_segment_is_not() {
        let state = crate::inference::pipeline::tests::make_test_state();
        state
            .split_models
            .insert((ModelId("m".into()), 0, 48), complete_entry(0, 48));
        let local = state.identity.node_id().clone();
        let exec = executor_for(state, vec![segment(local, (0, 24))]);
        assert_eq!(exec.local_whole_model_segment(), None);
    }

    /// The work is somewhere else — this is the remote fast path's case, and
    /// taking it here would run the model on the wrong machine.
    #[tokio::test]
    async fn a_segment_on_a_peer_is_not() {
        let state = crate::inference::pipeline::tests::make_test_state();
        state
            .split_models
            .insert((ModelId("m".into()), 0, 48), complete_entry(0, 48));
        let exec = executor_for(state, vec![segment(NodeId([9u8; 32]), (0, 48))]);
        assert_eq!(exec.local_whole_model_segment(), None);
    }

    /// Several segments means a real pipeline, whichever nodes they name.
    #[tokio::test]
    async fn a_multi_segment_plan_is_not() {
        let state = crate::inference::pipeline::tests::make_test_state();
        state
            .split_models
            .insert((ModelId("m".into()), 0, 48), complete_entry(0, 48));
        let local = state.identity.node_id().clone();
        let exec = executor_for(
            state,
            vec![segment(local.clone(), (0, 24)), segment(local, (24, 48))],
        );
        assert_eq!(exec.local_whole_model_segment(), None);
    }

    /// Images need the pipeline's own vision machinery.
    #[tokio::test]
    async fn a_request_carrying_an_image_is_not() {
        let state = crate::inference::pipeline::tests::make_test_state();
        state
            .split_models
            .insert((ModelId("m".into()), 0, 48), complete_entry(0, 48));
        let local = state.identity.node_id().clone();
        let mut exec = executor_for(state, vec![segment(local, (0, 48))]);
        exec.request.messages[0].images = vec![ImageData {
            rgb_bytes: vec![0, 0, 0],
            width: 1,
            height: 1,
        }];
        assert_eq!(exec.local_whole_model_segment(), None);
    }
}
