use crate::types::SamplingParams;

/// Apply temperature scaling to logits.
///
/// Higher temperature = more random, lower = more deterministic.
/// Temperature of 0 is treated as greedy (argmax).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 {
        return;
    }
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
}

/// Apply top-k filtering: keep only the k highest logits, set rest to -inf.
pub fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }

    // Find the k-th largest value
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k as usize - 1];

    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) sampling: keep the smallest set of tokens whose
/// cumulative probability exceeds p.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }

    // Convert to probabilities via softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let probs: Vec<f32> = probs.iter().map(|p| p / sum).collect();

    // Sort indices by probability descending
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find cutoff
    let mut cumulative = 0.0;
    let mut keep = std::collections::HashSet::new();
    for &idx in &indices {
        cumulative += probs[idx];
        keep.insert(idx);
        if cumulative >= p {
            break;
        }
    }

    // Mask out tokens not in the nucleus
    for (i, logit) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply frequency and presence penalties to logits based on token occurrence counts.
pub fn apply_repetition_penalties(
    logits: &mut [f32],
    token_counts: &[u32],
    params: &SamplingParams,
) {
    if params.frequency_penalty == 0.0 && params.presence_penalty == 0.0 {
        return;
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if let Some(&count) = token_counts.get(i) {
            if count > 0 {
                *logit -= params.frequency_penalty * count as f32;
                *logit -= params.presence_penalty;
            }
        }
    }
}

/// Sample a token index from logits using the given parameters.
///
/// Applies temperature, top-k, top-p in order, then samples from the
/// resulting distribution.
pub fn sample_token(logits: &mut [f32], params: &SamplingParams) -> u32 {
    // Greedy decoding when temperature is 0
    if params.temperature <= 0.0 {
        return argmax(logits);
    }

    apply_temperature(logits, params.temperature);
    apply_top_k(logits, params.top_k);
    apply_top_p(logits, params.top_p);

    // Softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = probs.iter().sum();

    // Weighted random selection
    let r: f32 = simple_random() * sum;
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            return i as u32;
        }
    }

    // Fallback
    (probs.len() - 1) as u32
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Simple pseudo-random float in [0, 1) using thread-local state.
/// Not cryptographically secure — fine for sampling.
fn simple_random() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let s = RandomState::new();
    let mut hasher = s.build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );
    (hasher.finish() as f32) / (u64::MAX as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 2.0).abs() < f32::EPSILON);
        assert!((logits[1] - 4.0).abs() < f32::EPSILON);
        assert!((logits[2] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn top_k_filtering() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        // Top-2 are indices 1 (5.0) and 4 (4.0)
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!((logits[1] - 5.0).abs() < f32::EPSILON);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert!((logits[4] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn greedy_sampling() {
        let mut logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample_token(&mut logits, &params);
        assert_eq!(token, 1); // Index of highest logit
    }

    #[test]
    fn argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 1.0]), 0);
    }
}
