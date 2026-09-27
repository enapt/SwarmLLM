//! Shared randomness for speculation across machines
//! (`docs/plans/split_speculation.md` Phase 4c).
//!
//! When the machine that drafts and the machine that samples draw their random
//! numbers from the SAME stream, a drafter whose distribution is close to the
//! target's draws the same token the target does — at any temperature. Each side
//! samples by Gumbel-max, `argmax_i(z_i / T + G_i)`, which is an exact sample of
//! that side's own distribution; with `G_i` shared, the two agree whenever the
//! draw lands on the same token, with probability at least `(1 - TV)/(1 + TV)`
//! (Daliri et al., arXiv 2408.07978, "Coupling without communication and
//! drafter-invariant speculative decoding"). A deterministic draft is accepted
//! only with probability p(draft), which at temperature 1.0 cost a 3-bit copy of
//! a 7B's far half 19 points of agreement (92.5% coupled vs 71.3%).
//!
//! **The noise is a pure function of (seed, absolute position, token id)** — a
//! counter-based generator, so either side computes any single `G_i` without
//! generating the ones before it, and no state has to agree beyond the seed.
//! Philox4x64-10 (Salmon et al., SC'11, "Parallel random numbers: as easy as 1, 2,
//! 3"): integer-only, so every machine produces bit-identical values. Pinned
//! against numpy's independent implementation and Random123's published
//! known-answer vector in the tests below.

/// Philox4x64 round multipliers and Weyl key increments (Random123).
const M0: u64 = 0xD2E7_470E_E14C_6C93;
const M1: u64 = 0xCA5A_8263_9512_1157;
const W0: u64 = 0x9E37_79B9_7F4A_7C15;
const W1: u64 = 0xBB67_AE85_84CA_A73B;

/// Separates this use of the seed from any other a future caller might make.
const DOMAIN: u64 = 0x5357_4152_4D4C_4C4D; // "SWARMLLM"

fn mulhilo(a: u64, b: u64) -> (u64, u64) {
    let p = (a as u128) * (b as u128);
    ((p >> 64) as u64, p as u64)
}

/// Philox4x64 with 10 rounds: 256 bits of output for a 256-bit counter under a
/// 128-bit key.
fn philox4x64_10(mut ctr: [u64; 4], mut key: [u64; 2]) -> [u64; 4] {
    for round in 0..10 {
        if round > 0 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
        let (hi0, lo0) = mulhilo(M0, ctr[0]);
        let (hi1, lo1) = mulhilo(M1, ctr[2]);
        ctr = [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0];
    }
    ctr
}

/// The largest Gumbel value [`CoupledNoise::gumbel`] can return: `u` is at least
/// 2^-53, so `E` is at least ~1.1e-16 and `G = -ln E` at most ~36.74.
pub(crate) const GUMBEL_MAX: f64 = 37.0;
/// The smallest: `u` is at most 1 - 2^-53, so `E` is at most ~36.74 and `G` at
/// least ~-3.60.
pub(crate) const GUMBEL_MIN: f64 = -3.65;

/// One request's shared noise stream. Both sides of a split hold the same seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoupledNoise {
    seed: u64,
}

impl CoupledNoise {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// A standard Gumbel sample for `token` at absolute sequence `position` — the
    /// position the sampled token will OCCUPY, never one relative to a draft,
    /// chunk or line (a relative key makes two sides disagree about which noise
    /// belongs to which token).
    ///
    /// `u` is taken from the top 52 bits with a half-step offset, so it is never 0
    /// or 1 (either would make `G` infinite) — 53 bits would round the largest
    /// value to exactly 1.0 — and `E = -ln(1 - u)` is computed with
    /// `ln_1p` so the upper tail keeps its precision: an f32 uniform would cut the
    /// Gumbel tail off near 17, which on a 152K vocabulary is not a rare event.
    pub fn gumbel(&self, position: u64, token: u32) -> f64 {
        let out = philox4x64_10([position, token as u64, 0, 0], [self.seed, DOMAIN]);
        let u = ((out[0] >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64);
        let e = -(-u).ln_1p();
        -e.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// numpy's `np.random.Philox` is an independent Philox4x64-10. It increments
    /// the counter BEFORE its first block, so numpy's `counter=c-1` is the block
    /// for `c` here. Values generated 2026-09-27 with numpy 1.26.
    #[test]
    fn philox_matches_numpys_independent_implementation() {
        let hex = |s: &str| u64::from_str_radix(s, 16).unwrap();
        let cases: [([u64; 4], [u64; 2], [&str; 4]); 3] = [
            (
                [7, 0, 0, 0],
                [42, 0],
                [
                    "d97b87792327f6f1",
                    "bd98f083584c2058",
                    "718641e5691cefc6",
                    "182ed409c4583e39",
                ],
            ),
            (
                [123, 456, 0, 0],
                [0xdead_beef, 0],
                [
                    "4ceea0d44641ed1f",
                    "e6361af9cdf0f9cb",
                    "9c4450fd5353e687",
                    "33e2e698e45177f9",
                ],
            ),
            (
                [5, 9, 0, 0],
                [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210],
                [
                    "42463582b1e74f6a",
                    "8e0d94aabceed3c9",
                    "f5682296b621012c",
                    "1a5204969aed662e",
                ],
            ),
        ];
        for (ctr, key, want) in cases {
            let got = philox4x64_10(ctr, key);
            let want: Vec<u64> = want.iter().map(|s| hex(s)).collect();
            assert_eq!(got.to_vec(), want, "ctr {ctr:?} key {key:?}");
        }
    }

    /// Random123's published known-answer vector for Philox4x64-10 at a zero
    /// counter and key.
    #[test]
    fn philox_matches_the_published_known_answer_vector() {
        assert_eq!(
            philox4x64_10([0; 4], [0; 2]),
            [
                0x1655_4d9e_ca36_314c,
                0xdb20_fe9d_672d_0fdc,
                0xd7e7_72ce_e186_176b,
                0x7e68_b68a_ec7b_a23b
            ]
        );
    }

    /// The noise is Gumbel(0, 1): mean = Euler-Mascheroni 0.5772, variance
    /// pi^2/6 = 1.6449 — and bounded by the constants the exact pruning relies on.
    #[test]
    fn the_noise_is_standard_gumbel_and_stays_inside_its_bounds() {
        let noise = CoupledNoise::new(0x5eed);
        let n = 200_000u32;
        let (mut sum, mut sq) = (0.0f64, 0.0f64);
        for i in 0..n {
            let g = noise.gumbel(u64::from(i / 1000), i % 1000);
            assert!((GUMBEL_MIN..=GUMBEL_MAX).contains(&g), "{g}");
            sum += g;
            sq += g * g;
        }
        let mean = sum / f64::from(n);
        let var = sq / f64::from(n) - mean * mean;
        assert!((mean - 0.5772).abs() < 0.01, "mean {mean}");
        assert!((var - 1.6449).abs() < 0.03, "variance {var}");
    }

    /// A pure function of (seed, position, token): the same inputs give the same
    /// value on every call and every machine, and each input changes it.
    #[test]
    fn the_noise_depends_on_exactly_seed_position_and_token() {
        let a = CoupledNoise::new(1);
        assert_eq!(a.gumbel(10, 20), a.gumbel(10, 20));
        assert_eq!(a.gumbel(10, 20), CoupledNoise::new(1).gumbel(10, 20));
        assert_ne!(a.gumbel(10, 20), a.gumbel(11, 20));
        assert_ne!(a.gumbel(10, 20), a.gumbel(10, 21));
        assert_ne!(a.gumbel(10, 20), CoupledNoise::new(2).gumbel(10, 20));
    }
}
