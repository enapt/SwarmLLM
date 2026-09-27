//! Guessing ahead with this model for a bigger one — the in-engine drafter of
//! speculation across computers (`pipeline::engine_drafter`, the worker's
//! `DaemonMsg::Draft`).

use crate::error::SwarmError;

use super::kv_cache::KvCacheStore;
use super::model::SplitModel;

impl SplitModel {
    /// Read `append` into this request's cache from position `keep`, then guess
    /// `gamma` tokens, reading each back in so the next can follow it — all but
    /// the last, which nothing has followed yet. The caller has already cut the
    /// cache back to `keep`; afterwards it holds `keep + append.len() + gamma - 1`
    /// positions.
    ///
    /// A guess is the token that will occupy position `keep + append.len() + j`.
    /// With `noise` it is drawn the way the target's sampler will draw that
    /// position (`sampling::sample_token_coupled`: the request's parameters,
    /// the reply so far in `history` plus the guesses before it, the shared
    /// noise keyed by the position); without, it is the plain argmax, as the
    /// llama.cpp drafter guesses (`pipeline::speculative::draft_next_gamma`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft_after(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        keep: usize,
        append: &[u32],
        gamma: usize,
        sampling: &crate::types::SamplingParams,
        history: &[u32],
        noise: Option<&crate::inference::coupled_noise::CoupledNoise>,
        prefill_chunk_tokens: usize,
    ) -> Result<Vec<u32>, SwarmError> {
        let first = keep + append.len();
        let input = self.tensor_from_ids(append)?;
        let mut logits = self.forward_prompt_in_chunks(
            &input,
            keep,
            kv,
            request_id,
            None,
            false,
            prefill_chunk_tokens,
        )?;
        let mut history = history.to_vec();
        let mut ctx = crate::inference::sampling::SamplingContext::new(0);
        let mut guesses = Vec::with_capacity(gamma);
        for j in 0..gamma {
            let mut row: Vec<f32> = logits
                .flatten_all()
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1())
                .map_err(SwarmError::internal)?;
            let t = match noise {
                Some(noise) => crate::inference::sampling::sample_token_coupled(
                    &mut row,
                    sampling,
                    &history,
                    &mut ctx,
                    noise,
                    (first + j) as u64,
                ),
                // The first of equal maxima, as every argmax here picks.
                None => {
                    row.iter()
                        .enumerate()
                        .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                            if x > best.1 {
                                (i, x)
                            } else {
                                best
                            }
                        })
                        .0 as u32
                }
            };
            guesses.push(t);
            history.push(t);
            if j + 1 < gamma {
                let next = self.token_tensor(t)?;
                logits = self.forward(&next, first + j, kv, request_id)?;
            }
        }
        Ok(guesses)
    }
}
