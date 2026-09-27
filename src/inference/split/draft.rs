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
    ///
    /// With `stop_below`, guessing stops early — after the guess that falls
    /// below it — once this model's own probability for a guess is under that
    /// figure: an unsure guess is the one the check is most likely to refuse,
    /// and every guess after it is guessed on top of it. Hugging Face's
    /// assisted generation does the same (`ConfidenceCriteria`: the softmax of
    /// the latest scores at the token just produced, compared with
    /// `assistant_confidence_threshold`, the token kept). The probability is
    /// read at the request's temperature (1 for a greedy request), before
    /// top-k and top-p.
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
        stop_below: Option<f32>,
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
            // Read before sampling, which may rewrite the row in place.
            let unsure = stop_below.map(|floor| (row.clone(), floor));
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
            if let Some((raw, floor)) = unsure {
                if probability_at(&raw, t as usize, sampling.temperature) < floor {
                    break;
                }
            }
            if j + 1 < gamma {
                let next = self.token_tensor(t)?;
                logits = self.forward(&next, first + j, kv, request_id)?;
            }
        }
        Ok(guesses)
    }
}

/// `softmax(logits / T)[token]`, with `T` the request's temperature or 1 for a
/// greedy one — the probability a drafter gives its own guess.
fn probability_at(logits: &[f32], token: usize, temperature: f32) -> f32 {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let sum: f64 = logits.iter().map(|&x| f64::from((x - max) / t).exp()).sum();
    let own = logits
        .get(token)
        .map_or(f64::NEG_INFINITY, |&x| f64::from((x - max) / t));
    (own.exp() / sum) as f32
}

#[cfg(test)]
mod tests {
    use super::probability_at;

    #[test]
    fn a_guess_probability_is_the_softmax_at_the_requests_temperature() {
        let logits = [2.0f32, 1.0, 0.0];
        let p = probability_at(&logits, 0, 0.0);
        let e = [2.0f64.exp(), 1.0f64.exp(), 1.0];
        assert!((f64::from(p) - e[0] / (e[0] + e[1] + e[2])).abs() < 1e-6);
        // A higher temperature flattens it.
        assert!(probability_at(&logits, 0, 2.0) < p);
        assert_eq!(probability_at(&[f32::NEG_INFINITY; 3], 0, 1.0), 0.0);
    }
}
