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
        let (prompt, sampling) = self
            .build_prompt_and_stops(self.request.sampling_params.clone())
            .await;

        tracing::info!(
            %request_id,
            model = %model_id,
            layer_start = layer_range.0,
            layer_end = layer_range.1,
            "DIAG: single-segment plan names this node — running it as a local generate"
        );

        // `session_id` is carried through so a multi-turn conversation keeps
        // its KV entry, exactly as it does on the API fast path.
        let out = self
            .shared_state
            .model_process_pool
            .generate(
                &model_id,
                layer_range,
                prompt,
                sampling,
                request_id,
                self.request.session_id.clone(),
                token_tx,
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
        let meta =
            crate::api::openai::get_split_model_meta(&self.shared_state, &self.request.model_id)?;
        (meta.layer_range == segment.layer_range).then_some(meta.layer_range)
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
