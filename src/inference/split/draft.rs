//! Guessing ahead with this model for a bigger one — the in-engine drafter of
//! speculation across computers (`pipeline::engine_drafter`, the worker's
//! `DaemonMsg::Draft`).

use crate::error::SwarmError;

use super::kv_cache::KvCacheStore;
use super::model::SplitModel;

/// The longest read a draft call makes as single positions on a card (see
/// `draft_after`): past a few positions, one pass reading the weights once
/// wins even against captured steps.
const SINGLE_STEP_READ_MAX: usize = 4;

impl SplitModel {
    /// Read `append` into this request's cache from position `keep`, then guess
    /// `gamma` tokens, reading each back in so the next can follow it — all but
    /// the last, which nothing has followed yet. The cache is cut back to `keep`
    /// first; afterwards it holds `keep + append.len() + gamma - 1` positions.
    ///
    /// A cache holding FEWER than `keep` positions is refused, not read on top
    /// of: the call continues a context this worker no longer has — the entry
    /// expired while other rounds drafted, or the worker was replaced — and
    /// reading `append` at position `keep` over it would guess from a context
    /// that was never written, with nothing failing. Refused, the reply goes
    /// on without guessing (`pipeline::dsd`, `drafting_off`).
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
        let held = kv.request_positions(self.kv_model_key(), request_id);
        if held < keep {
            return Err(SwarmError::ServiceUnavailable(format!(
                "the guessing model holds {held} positions of this reply, not the {keep} \
                 this call continues from — its cache expired or its worker was replaced"
            )));
        }
        // Also empties a first read's cache (`keep` 0) of whatever an earlier
        // call that failed part-way left in it.
        kv.truncate_request_to(self.kv_model_key(), request_id, keep)?;
        let first = keep + append.len();
        // A short read that continues the reply goes through as single
        // positions on a card: each is a decode step, sent as a CUDA graph at
        // ~3 ms on a 0.5B, where ONE multi-position forward of the same 2-4
        // tokens takes the uncaptured prompt path at 12-17 ms (measured on the
        // split rig, 2026-09-30). Most calls read two — the guess the last call
        // never fed back, and the check's own token — so this was most of a
        // call's cost. On the processor one pass reads the weights once and
        // stays cheaper; the prompt's first read is a prompt pass either way.
        let mut logits =
            if keep > 0 && append.len() <= SINGLE_STEP_READ_MAX && self.runs_entirely_on_card() {
                let mut last = None;
                for (i, &id) in append.iter().enumerate() {
                    let step = self.token_tensor(id)?;
                    last = Some(self.forward(&step, keep + i, kv, request_id)?);
                }
                last.expect("append is not empty — the worker refuses an empty draft call")
            } else {
                let input = self.tensor_from_ids(append)?;
                self.forward_prompt_in_chunks(
                    &input,
                    keep,
                    kv,
                    request_id,
                    None,
                    false,
                    prefill_chunk_tokens,
                )?
            };
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
