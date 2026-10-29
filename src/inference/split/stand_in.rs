//! A stand-in for the run of layers a card's guess skips, fitted while the
//! reply is written (2026-10-28, FUTURE_WORK #366).
//!
//! A whole model split between the card and the processor guesses on the card
//! alone by skipping the processor's run (`SplitModel::card_guess_skips`).
//! Skipped with nothing in its place, the run is taken to add nothing to the
//! hidden state. But every real forward of the same reply — each plain step,
//! each check — computes the run exactly, so the hidden state where the run
//! starts and where it ends are there to be kept at no cost. [`RunStandIn`]
//! keeps them for the last [`STAND_IN_WINDOW`] positions and answers with a
//! [`SkipBridge::Recall`]: linear kernel ridge from the normalised start `φ`
//! to the run's effect `Δ`, the arithmetic `examples/skip_agreement_probe.rs`
//! measured as `recall@0.3`. Tokens kept a round of four, nothing in the gap →
//! this stand-in: with 14 of 32 skipped from layer 3, Mistral 7B 0.34 → 1.73
//! and Llama 3.1 8B 0.79 → 1.12; with 25 of 32 from layer 4, 0.03 → 0.55 and
//! 0.03 → 0.47; Qwen 2.5 14B 1.54 → 1.76 with 21 of 48 skipped from layer 5,
//! 0.26 → 0.62 with 33 (2026-10-28).
//!
//! The probe refitted from scratch at every position — `(K + λI)⁻¹` of a
//! 256 × 256 Gram matrix, tens of milliseconds on the host. Here the inverse
//! is kept current a row at a time ([`WindowInverse`]): a new position adds a
//! row and a column by the block inverse, the oldest leaves by the matching
//! downdate, each O(m²) — the sliding-window kernel RLS of Van Vaerenbergh,
//! Vía and Santamaría (ICASSP 2006). It is rebuilt whole once per window's
//! worth of updates, and whenever an update's pivot says the arithmetic has
//! drifted.
//!
//! **A model carries what its earlier replies learned** ([`StandInPrior`],
//! 2026-10-29): the last [`PRIOR_ROWS`] positions of the conversations it
//! served before, as a second kernel ridge fitted once a reply is out, which
//! the reply's window then corrects — fitted to what the prior leaves of each
//! effect (`examples/skip_agreement_probe.rs`, `prior2048+recall@0.3`).
//! Tokens kept a round over six short conversations after a long one,
//! window → window over the prior: Qwen 2.5 14B 0.48 → 0.64 with 33 of 48
//! skipped, Llama 3.1 8B 0.32 → 0.53 with 25 of 32 (rounds 12-13; every
//! earlier position, uncapped, 0.66 and 0.55). A prompt longer than the window
//! joins the prior for its own reply first ([`StandInPrior::fit_for_prompt`]):
//! the long conversation itself, 0.66 → 1.17 and 0.69 → 1.23.

use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device, Tensor};

use super::SkipBridge;
use crate::error::SwarmError;

/// Positions a stand-in is fitted from: the most recent, prompt included.
pub const STAND_IN_WINDOW: usize = 256;

/// The ridge: `recall@0.3`, the probe's best (`recall@3` was worse on all
/// three models).
pub const STAND_IN_LAMBDA: f64 = 0.3;

/// Positions before a stand-in is offered: fewer is a fit to noise.
const LEAST_ROWS: usize = 8;

/// Positions of a model's earlier conversations its prior is fitted from,
/// the most recent kept. Round 13 (FUTURE_WORK #366 (1)), tokens kept a round
/// over six short conversations by how many: Qwen 2.5 14B 0.60 at 512, 0.60
/// at 1,024, 0.64 at 2,048, 0.66 every one; Llama 3.1 8B 0.48, 0.50, 0.53,
/// 0.55. On the card it is `2 × PRIOR_ROWS × hidden` f32 — 84 MB for a
/// 5,120-wide model — and each guess reads it once.
pub const PRIOR_ROWS: usize = 2048;

/// The share of the mean of a stand-in's misses at the last
/// [`STAND_IN_CARRY_POSITIONS`] kept positions the reply's card guesses carry
/// ([`RunStandIn::with_carry`], round 15's `carried~0.5x4`).
pub const STAND_IN_CARRY: f64 = 0.5;
pub const STAND_IN_CARRY_POSITIONS: usize = 4;

/// The most Newton–Schulz steps [`ridge_inverse`] takes. From `I / ‖A‖∞` the
/// residual's largest eigenvalue squares each step, so the slow first phase
/// takes about `log₂(‖A‖∞ / λ)` steps — 13 for 2,048 unit rows at λ 0.3 —
/// and a few more reach f32's floor.
const RIDGE_INVERSE_MOST_STEPS: usize = 40;

/// The largest entry of `A·X − I` an inverse is accepted at.
const RIDGE_INVERSE_GOOD_ENOUGH: f32 = 1e-2;

/// The most new positions added a row at a time; more (a prompt) rebuilds.
const MOST_ROWS_ADDED_IN_STEPS: usize = 16;

/// `(K + λI)⁻¹` for the Gram matrix `K` of a window of unit rows, in f64 on
/// the host, row-major: grown by a row with [`Self::push`], shrunk from the
/// front with [`Self::pop_front`].
#[derive(Debug, Clone)]
pub(crate) struct WindowInverse {
    m: usize,
    a: Vec<f64>,
    lambda: f64,
}

impl WindowInverse {
    /// The inverse of `gram + λI` (`gram` `m × m`, row-major) by Cholesky;
    /// `None` where the matrix is not positive definite — not finite, or a
    /// ridge too small for the rounding.
    pub(crate) fn from_gram(gram: &[f64], m: usize, lambda: f64) -> Option<Self> {
        debug_assert_eq!(gram.len(), m * m);
        // L, lower: L·Lᵀ = gram + λI.
        let mut l = vec![0.0f64; m * m];
        for j in 0..m {
            let mut d = gram[j * m + j] + lambda;
            for k in 0..j {
                d -= l[j * m + k] * l[j * m + k];
            }
            if !(d.is_finite() && d > 0.0) {
                return None;
            }
            let d = d.sqrt();
            l[j * m + j] = d;
            for i in j + 1..m {
                let mut s = gram[i * m + j];
                for k in 0..j {
                    s -= l[i * m + k] * l[j * m + k];
                }
                l[i * m + j] = s / d;
            }
        }
        // M = L⁻¹, lower, column by column.
        let mut inv_l = vec![0.0f64; m * m];
        for c in 0..m {
            inv_l[c * m + c] = 1.0 / l[c * m + c];
            for i in c + 1..m {
                let mut s = 0.0;
                for k in c..i {
                    s += l[i * m + k] * inv_l[k * m + c];
                }
                inv_l[i * m + c] = -s / l[i * m + i];
            }
        }
        // (L·Lᵀ)⁻¹ = Mᵀ·M.
        let mut a = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..=i {
                let mut s = 0.0;
                for k in i..m {
                    s += inv_l[k * m + i] * inv_l[k * m + j];
                }
                a[i * m + j] = s;
                a[j * m + i] = s;
            }
        }
        Some(Self { m, a, lambda })
    }

    pub(crate) fn len(&self) -> usize {
        self.m
    }

    /// Add a row whose products with the window's rows are `k` (`m`) and
    /// with itself `c`. False where the update's pivot — `c + λ − kᵀ·A·k`,
    /// at least λ in exact arithmetic — says the inverse has drifted: the
    /// caller rebuilds it.
    pub(crate) fn push(&mut self, k: &[f64], c: f64) -> bool {
        let m = self.m;
        debug_assert_eq!(k.len(), m);
        let u: Vec<f64> = (0..m)
            .map(|i| {
                self.a[i * m..(i + 1) * m]
                    .iter()
                    .zip(k)
                    .map(|(x, y)| x * y)
                    .sum()
            })
            .collect();
        let s = c + self.lambda - k.iter().zip(&u).map(|(x, y)| x * y).sum::<f64>();
        if !(s.is_finite() && s > 0.5 * self.lambda) {
            return false;
        }
        let n = m + 1;
        let mut a = vec![0.0f64; n * n];
        for i in 0..m {
            let ui = u[i] / s;
            for j in 0..m {
                a[i * n + j] = self.a[i * m + j] + ui * u[j];
            }
            a[i * n + m] = -ui;
            a[m * n + i] = -ui;
        }
        a[m * n + m] = 1.0 / s;
        self.a = a;
        self.m = n;
        true
    }

    /// Drop the oldest row: `A₁₁ − a₀·a₀ᵀ / a₀₀` of the current inverse
    /// partitioned at it. False where its pivot is not positive.
    pub(crate) fn pop_front(&mut self) -> bool {
        let m = self.m;
        if m == 0 {
            return false;
        }
        let b00 = self.a[0];
        if !(b00.is_finite() && b00 > 0.0) {
            return false;
        }
        let n = m - 1;
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            let bi = self.a[(i + 1) * m] / b00;
            for j in 0..n {
                a[i * n + j] = self.a[(i + 1) * m + j + 1] - bi * self.a[j + 1];
            }
        }
        self.a = a;
        self.m = n;
        true
    }

    fn as_f32(&self) -> Vec<f32> {
        self.a.iter().map(|&v| v as f32).collect()
    }
}

/// What a skipped run did at the reply's real positions, and the
/// [`SkipBridge`] fitted from it (module docs).
pub struct RunStandIn {
    run_start: usize,
    run_end: usize,
    capture: HashSet<usize>,
    device: Device,
    window: usize,
    lambda: f64,
    /// `φ`, the window's normalised hidden states where the run starts,
    /// `[n, hidden]` f32 on `device`.
    phi: Option<Tensor>,
    /// `Δ`, what the run added to them, `[n, hidden]`.
    delta: Option<Tensor>,
    /// `Σ Δ` and `Σ φ` over every position seen, and their count: the
    /// bridge's base is the mean effect, as the probe's was — less the prior's
    /// estimate at the mean φ where there is one (affine, so the mean of its
    /// residuals).
    delta_sum: Option<Tensor>,
    phi_sum: Option<Tensor>,
    seen: usize,
    /// The model's earlier conversations ([`StandInPrior`]); the window is
    /// fitted to what it leaves of each effect. Set before anything is learned.
    prior: Option<PriorFit>,
    /// Every position learned, `(φ, Δ)` on the host in arrival order, the
    /// oldest dropped past [`PRIOR_ROWS`] — what [`StandInPrior::absorb`]
    /// takes at the reply's end; kept only where [`Self::collecting`] said.
    collect: bool,
    taken: std::collections::VecDeque<(Tensor, Tensor)>,
    taken_rows: usize,
    inverse: Option<WindowInverse>,
    /// Rows added since the inverse was last rebuilt whole.
    since_rebuild: usize,
    bridge: Option<SkipBridge>,
    /// Inverses rebuilt whole, and the milliseconds the stand-in has cost.
    pub rebuilds: u64,
    pub ms: f64,
    /// Positions refused for a value that was not finite: one such row would
    /// leave the window without an inverse until it slid out, and a model's
    /// prior without a fit until 2,048 more had come (gotcha #828 saw a card
    /// hand back a NaN row).
    pub not_finite: u64,
    /// The share of its recent misses added to the next guesses, and over
    /// how many positions ([`Self::with_carry`]); a share of 0 adds none.
    carry_share: f64,
    carry_positions: usize,
    /// Its misses at the last `carry_positions` positions checks kept,
    /// oldest first, `[hidden]` each.
    misses: std::collections::VecDeque<Tensor>,
    /// `carry_share` of their mean, `[hidden]`: added to the bridge's base
    /// at each [`Self::refresh`].
    carry: Option<Tensor>,
}

impl RunStandIn {
    /// A stand-in for the one run of layers `skips` names, made on `device`;
    /// `None` unless the run is a single block that starts after layer 0
    /// (its start is the hidden state after the layer before it) and ends
    /// before the last.
    pub fn for_run(skips: &[bool], device: &Device) -> Option<Self> {
        let run_start = skips.iter().position(|&s| s)?;
        let run_end = run_start + skips[run_start..].iter().take_while(|&&s| s).count();
        if run_start == 0 || run_end >= skips.len() || skips[run_end..].iter().any(|&s| s) {
            return None;
        }
        Some(Self::new(
            run_start,
            run_end,
            [run_start - 1, run_end - 1].into_iter().collect(),
            device,
        ))
    }

    /// A stand-in for the FAR run `[run_start, run_end)` of a chain whose two
    /// ends this node holds — its first segment ends at `run_start`, its last
    /// begins at `run_end` ("Start and finish on this computer"). It learns
    /// from the hidden state the first segment hands on and the one the last
    /// segment is handed back ([`Self::learn_pairs`]), so it needs none of the
    /// far layers' weights (`docs/plans/split_speculation.md` § "Guess at the
    /// head through a stand-in for the far layers"). It stands in for layers,
    /// never for memory: a guess through it runs the last segment on the
    /// request's OWN cache (gotcha #508). `None` for an empty run or one
    /// starting at layer 0.
    pub fn for_far_run(run_start: usize, run_end: usize, device: &Device) -> Option<Self> {
        (run_start > 0 && run_end > run_start)
            .then(|| Self::new(run_start, run_end, HashSet::new(), device))
    }

    fn new(run_start: usize, run_end: usize, capture: HashSet<usize>, device: &Device) -> Self {
        Self {
            run_start,
            run_end,
            capture,
            device: device.clone(),
            window: STAND_IN_WINDOW,
            lambda: STAND_IN_LAMBDA,
            phi: None,
            delta: None,
            delta_sum: None,
            phi_sum: None,
            seen: 0,
            prior: None,
            collect: false,
            taken: std::collections::VecDeque::new(),
            taken_rows: 0,
            inverse: None,
            since_rebuild: 0,
            bridge: None,
            rebuilds: 0,
            ms: 0.0,
            not_finite: 0,
            carry_share: 0.0,
            carry_positions: 1,
            misses: std::collections::VecDeque::new(),
            carry: None,
        }
    }

    /// Add `share` of the mean of the stand-in's misses at the last
    /// `positions` positions checks kept — what the run really added there,
    /// less the bridge's estimate before they were learned — to the next
    /// guesses' estimate: a miss is much like the ones before it. Round 15
    /// (`skip_agreement_probe`'s `carried~ρxk`, tokens kept a round): half
    /// the mean of the last four kept 0.90 where the stand-in alone kept
    /// 0.72 (Qwen2.5-14B, 33 of 48 skipped) and 0.63 where it kept 0.63
    /// (Llama 3.1 8B, 25 of 32); half the last one alone 0.96 and 0.53 — one
    /// miss is too noisy a guide for some models; the whole of it
    /// overshoots (0.62, 0.31). No prior art found (closest: CLaSp, arXiv
    /// 2505.24196, which re-picks the skipped layers from the last check's
    /// hidden states).
    pub fn with_carry(mut self, share: f64, positions: usize) -> Self {
        self.carry_share = share;
        self.carry_positions = positions.max(1);
        self
    }

    /// The same, fitted from the last `window` positions instead of
    /// [`STAND_IN_WINDOW`] (the probe's `PROBE_WINDOW`).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(LEAST_ROWS);
        self
    }

    /// Keep every position learned for the model's prior
    /// ([`StandInPrior::absorb`]), and capture a prompt's last [`PRIOR_ROWS`]
    /// for it ([`Self::prompt_capture`]).
    pub fn collecting(mut self) -> Self {
        self.collect = true;
        self
    }

    /// Fit the window to what `prior` leaves of each effect; `None` fits it
    /// to the effects. Refused once a position has been learned: the window's
    /// effects would be measured against two different priors.
    pub fn set_prior(&mut self, prior: Option<PriorFit>) -> Result<(), SwarmError> {
        if self.seen > 0 {
            return Err(SwarmError::Internal(
                "stand-in: a prior is set before anything is learned".into(),
            ));
        }
        self.prior = prior;
        self.bridge = None;
        Ok(())
    }

    /// Positions of earlier conversations its prior was fitted from; 0
    /// without one.
    pub fn prior_rows(&self) -> usize {
        self.prior.as_ref().map_or(0, PriorFit::rows)
    }

    /// Whether it keeps what it learns for the model's prior
    /// ([`Self::collecting`]).
    pub fn collects(&self) -> bool {
        self.collect
    }

    /// The positions a prompt pass captures for it (`split::Capture::last`):
    /// its window, or the prior's rows where it is collecting for one.
    pub fn prompt_capture(&self) -> usize {
        if self.collect {
            self.window.max(PRIOR_ROWS)
        } else {
            self.window
        }
    }

    /// The run it stands in for: its first layer and the layer past its last.
    pub fn run(&self) -> (usize, usize) {
        (self.run_start, self.run_end)
    }

    /// The layers a forward captures for [`Self::learn`]: the one before the
    /// run and the run's last.
    pub fn capture_layers(&self) -> &HashSet<usize> {
        &self.capture
    }

    /// The most positions it is fitted from.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Positions it is fitted from now.
    pub fn rows(&self) -> usize {
        self.inverse.as_ref().map_or(0, WindowInverse::len)
    }

    /// Learn from the first `kept` positions of a forward that captured
    /// [`Self::capture_layers`] (`[1, positions, hidden]` each): those its
    /// cache keeps — a check's accepted guesses and the token after them.
    pub fn learn(
        &mut self,
        caught: &HashMap<usize, Tensor>,
        kept: usize,
    ) -> Result<(), SwarmError> {
        if kept == 0 {
            return Ok(());
        }
        let rows = self.rows_of(caught, kept)?;
        self.learn_kept(rows)
    }

    /// [`Self::learn`] from a chain's two ends ([`Self::for_far_run`]): `h_in`
    /// the first segment's output at positions a check kept, `h_out` the last
    /// segment's input at the same positions (`[kept, hidden]` or
    /// `[1, kept, hidden]`, on any device).
    pub fn learn_pairs(&mut self, h_in: &Tensor, h_out: &Tensor) -> Result<(), SwarmError> {
        let rows = self.pair_rows(h_in, h_out)?;
        if rows.0.dim(0).map_err(SwarmError::internal)? == 0 {
            return Ok(());
        }
        self.learn_kept(rows)
    }

    /// Kept positions' `(φ, Δ)` into the window, their miss measured first.
    fn learn_kept(&mut self, (phi, delta): (Tensor, Tensor)) -> Result<(), SwarmError> {
        let started = std::time::Instant::now();
        let out = self
            .note_miss(&phi, &delta)
            .and_then(|()| self.learn_inner(phi, delta));
        self.ms += started.elapsed().as_secs_f64() * 1000.0;
        out
    }

    /// The carry for the next guesses ([`Self::with_carry`]): the bridge's
    /// misses at the last of a check's kept positions `(phi, delta)`, before
    /// they are learned, measured without the carry the bridge already
    /// held, join the recent ones. Without a bridge there is no miss to
    /// measure; misses that are not finite are left out, as their rows are.
    fn note_miss(&mut self, phi: &Tensor, delta: &Tensor) -> Result<(), SwarmError> {
        if self.carry_share == 0.0 {
            return Ok(());
        }
        let Some(bridge) = &self.bridge else {
            return Ok(());
        };
        let misses = (|| -> candle_core::Result<Option<Tensor>> {
            let n = phi.dim(0)?;
            let k = n.min(self.carry_positions);
            let Some(est) = bridge.effect(&phi.narrow(0, n - k, k)?)? else {
                return Ok(None);
            };
            let est = match &self.carry {
                Some(held) => est.broadcast_sub(held)?,
                None => est,
            };
            Ok(Some((delta.narrow(0, n - k, k)? - est)?))
        })()
        .map_err(SwarmError::internal)?;
        let Some(misses) = misses.filter(|m| all_finite(m).unwrap_or(false)) else {
            return Ok(());
        };
        (|| -> candle_core::Result<()> {
            for i in 0..misses.dim(0)? {
                self.misses.push_back(misses.get(i)?);
            }
            while self.misses.len() > self.carry_positions {
                self.misses.pop_front();
            }
            let held: Vec<&Tensor> = self.misses.iter().collect();
            self.carry = Some((Tensor::stack(&held, 0)?.mean(0)? * self.carry_share)?);
            Ok(())
        })()
        .map_err(SwarmError::internal)
    }

    /// [`Self::learn`] of rows [`Self::rows_of`] made: where a prompt's rows
    /// are wanted for the prior's fit ([`StandInPrior::fit_for_prompt`])
    /// before [`Self::set_prior`].
    pub fn learn_rows(&mut self, phi: Tensor, delta: Tensor) -> Result<(), SwarmError> {
        if phi.dim(0).map_err(SwarmError::internal)? == 0 {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let out = self.learn_inner(phi, delta);
        self.ms += started.elapsed().as_secs_f64() * 1000.0;
        out
    }

    /// `(φ, Δ)` of the first `kept` positions of a forward that captured
    /// [`Self::capture_layers`]: the normalised hidden state where the run
    /// starts and what the run added to it, `[kept, hidden]` f32 on its device.
    pub fn rows_of(
        &self,
        caught: &HashMap<usize, Tensor>,
        kept: usize,
    ) -> Result<(Tensor, Tensor), SwarmError> {
        let take = |layer: usize| -> Result<Tensor, SwarmError> {
            let t = caught.get(&layer).ok_or_else(|| {
                SwarmError::Internal(format!("stand-in: layer {layer} was not captured"))
            })?;
            t.squeeze(0)
                .and_then(|t| t.narrow(0, 0, kept))
                .and_then(|t| t.to_device(&self.device))
                .and_then(|t| t.to_dtype(DType::F32))
                .map_err(SwarmError::internal)
        };
        let h_in = take(self.run_start - 1)?;
        let h_out = take(self.run_end - 1)?;
        Self::phi_delta(&h_in, &h_out)
    }

    /// `(φ, Δ)` of a chain's paired rows ([`Self::learn_pairs`]): what
    /// [`Self::rows_of`] makes of a captured forward, from the hidden states
    /// the chain's two ends hold — `[rows, hidden]` or `[1, rows, hidden]`.
    pub fn pair_rows(&self, h_in: &Tensor, h_out: &Tensor) -> Result<(Tensor, Tensor), SwarmError> {
        let flat = |t: &Tensor| -> Result<Tensor, SwarmError> {
            let t = if t.rank() == 3 {
                t.squeeze(0)
            } else {
                Ok(t.clone())
            };
            t.and_then(|t| t.to_device(&self.device))
                .and_then(|t| t.to_dtype(DType::F32))
                .map_err(SwarmError::internal)
        };
        let (h_in, h_out) = (flat(h_in)?, flat(h_out)?);
        if h_in.dims() != h_out.dims() {
            return Err(SwarmError::Internal(format!(
                "stand-in: paired rows {:?} and {:?} differ",
                h_in.dims(),
                h_out.dims()
            )));
        }
        Self::phi_delta(&h_in, &h_out)
    }

    /// The normalised start and the run's effect: the ONE arithmetic of a
    /// stand-in's rows, whichever way they were caught.
    fn phi_delta(h_in: &Tensor, h_out: &Tensor) -> Result<(Tensor, Tensor), SwarmError> {
        (|| -> candle_core::Result<(Tensor, Tensor)> {
            let delta = (h_out - h_in)?;
            let norm = h_in.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-6, f64::MAX)?;
            Ok((h_in.broadcast_div(&norm)?, delta))
        })()
        .map_err(SwarmError::internal)
    }

    fn learn_inner(&mut self, x_phi: Tensor, x_delta: Tensor) -> Result<(), SwarmError> {
        let kept = x_phi.dim(0).map_err(SwarmError::internal)?;
        if !(all_finite(&x_phi)? && all_finite(&x_delta)?) {
            self.not_finite += kept as u64;
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    positions = kept,
                    "card guesses: a forward's hidden state was not finite; its positions are left out of the stand-in"
                )
            });
            return Ok(());
        }
        let add = |old: Option<Tensor>, new: Tensor| -> Result<Tensor, SwarmError> {
            match old {
                Some(s) => (s + new).map_err(SwarmError::internal),
                None => Ok(new),
            }
        };
        let (d_sum, p_sum) = (
            x_delta.sum(0).map_err(SwarmError::internal)?,
            x_phi.sum(0).map_err(SwarmError::internal)?,
        );
        self.delta_sum = Some(add(self.delta_sum.take(), d_sum)?);
        self.phi_sum = Some(add(self.phi_sum.take(), p_sum)?);
        self.seen += kept;
        if self.collect {
            let host = |t: &Tensor| t.to_device(&Device::Cpu).map_err(SwarmError::internal);
            self.taken.push_back((host(&x_phi)?, host(&x_delta)?));
            self.taken_rows += kept;
            while let Some((front, _)) = self.taken.front() {
                let rows = front.dim(0).map_err(SwarmError::internal)?;
                if self.taken_rows - rows < PRIOR_ROWS {
                    break;
                }
                self.taken_rows -= rows;
                self.taken.pop_front();
            }
        }
        // Only the window's rows go on — measured against the prior where
        // there is one.
        let tail = kept.min(self.window);
        let narrow = |t: Tensor| -> Result<Tensor, SwarmError> {
            if tail == kept {
                return Ok(t);
            }
            t.narrow(0, kept - tail, tail)
                .and_then(|t| t.contiguous())
                .map_err(SwarmError::internal)
        };
        let (x_phi, x_delta) = (narrow(x_phi)?, narrow(x_delta)?);
        let x_delta = match &self.prior {
            Some(prior) => (x_delta - prior.estimate(&x_phi)?).map_err(SwarmError::internal)?,
            None => x_delta,
        };
        let kept = tail;

        let held = self
            .phi
            .as_ref()
            .map_or(Ok(0), |p| p.dim(0))
            .map_err(SwarmError::internal)?;
        let stepwise = kept <= MOST_ROWS_ADDED_IN_STEPS && self.since_rebuild + kept < self.window;
        let mut drifted = true;
        if let (true, Some(inverse), Some(phi)) = (stepwise, self.inverse.as_mut(), &self.phi) {
            drifted = false;
            // Each new row's products with the rows before it: the window,
            // then the new rows ahead of it.
            let products: Vec<Vec<f32>> = Tensor::cat(&[phi, &x_phi], 0)
                .and_then(|all| x_phi.matmul(&all.t()?))
                .and_then(|p| p.to_vec2())
                .map_err(SwarmError::internal)?;
            let mut dropped = 0usize;
            for (j, row) in products.iter().enumerate() {
                if inverse.len() >= self.window {
                    if !inverse.pop_front() {
                        drifted = true;
                        break;
                    }
                    dropped += 1;
                }
                let k: Vec<f64> = row[dropped..held + j]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                if !inverse.push(&k, f64::from(row[held + j])) {
                    drifted = true;
                    break;
                }
            }
            self.since_rebuild += kept;
        }
        // The window on the card, newest last.
        let keep_last = |old: Option<Tensor>, new: &Tensor| -> Result<Tensor, SwarmError> {
            let joined = match old {
                Some(o) => Tensor::cat(&[&o, new], 0),
                None => Ok(new.clone()),
            }
            .map_err(SwarmError::internal)?;
            let n = joined.dim(0).map_err(SwarmError::internal)?;
            let from = n.saturating_sub(self.window);
            joined
                .narrow(0, from, n - from)
                .map_err(SwarmError::internal)
        };
        self.phi = Some(keep_last(self.phi.take(), &x_phi)?);
        self.delta = Some(keep_last(self.delta.take(), &x_delta)?);
        if drifted {
            self.rebuild()?;
        }
        self.bridge = None;
        Ok(())
    }

    /// The inverse made whole from the window's Gram matrix.
    fn rebuild(&mut self) -> Result<(), SwarmError> {
        let Some(phi) = &self.phi else {
            return Ok(());
        };
        let m = phi.dim(0).map_err(SwarmError::internal)?;
        let gram: Vec<f64> = phi
            .matmul(&phi.t().map_err(SwarmError::internal)?)
            .and_then(|g| g.flatten_all())
            .and_then(|g| g.to_vec1::<f32>())
            .map_err(SwarmError::internal)?
            .into_iter()
            .map(f64::from)
            .collect();
        self.inverse = WindowInverse::from_gram(&gram, m, self.lambda);
        self.since_rebuild = 0;
        self.rebuilds += 1;
        Ok(())
    }

    /// Fit the stand-in to the window as it is now, where anything was
    /// learned since the last fit; [`Self::bridge`] reads it.
    pub fn refresh(&mut self) -> Result<(), SwarmError> {
        if self.bridge.is_none() {
            let started = std::time::Instant::now();
            self.bridge = self.fit()?;
            if let (Some(SkipBridge::Recall { base, .. }), Some(carry)) =
                (self.bridge.as_mut(), &self.carry)
            {
                *base = (&*base + carry).map_err(SwarmError::internal)?;
            }
            self.ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(())
    }

    /// The stand-in as last [refreshed](Self::refresh); `None` before
    /// [`LEAST_ROWS`] positions, or where the inverse could not be made.
    pub fn bridge(&self) -> Option<&SkipBridge> {
        self.bridge.as_ref()
    }

    fn fit(&self) -> Result<Option<SkipBridge>, SwarmError> {
        let (Some(inverse), Some(phi), Some(delta), Some(sum), Some(phi_sum)) = (
            &self.inverse,
            &self.phi,
            &self.delta,
            &self.delta_sum,
            &self.phi_sum,
        ) else {
            return Ok(None);
        };
        let m = inverse.len();
        if m < LEAST_ROWS || self.seen == 0 {
            return Ok(None);
        }
        let mean_effect = (sum / self.seen as f64).map_err(SwarmError::internal)?;
        // The window's effects are what the prior left: their mean is the
        // mean effect less the prior's estimate at the mean φ.
        let (left_mean, base) = match &self.prior {
            Some(prior) => {
                let mean_phi = (phi_sum / self.seen as f64)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(SwarmError::internal)?;
                let at_mean = prior
                    .estimate(&mean_phi)?
                    .squeeze(0)
                    .map_err(SwarmError::internal)?;
                let left = (&mean_effect - at_mean).map_err(SwarmError::internal)?;
                let base = (&prior.base + &left).map_err(SwarmError::internal)?;
                (left, base)
            }
            None => (mean_effect.clone(), mean_effect),
        };
        (|| -> candle_core::Result<Option<SkipBridge>> {
            let inv = Tensor::from_vec(inverse.as_f32(), (m, m), &self.device)?;
            let weights = inv.matmul(&delta.broadcast_sub(&left_mean)?.contiguous()?)?;
            Ok(Some(SkipBridge::Recall {
                run_end: self.run_end,
                base,
                features: phi.contiguous()?,
                weights,
                prior: self
                    .prior
                    .as_ref()
                    .map(|p| (p.features.clone(), p.weights.clone())),
            }))
        })()
        .map_err(SwarmError::internal)
    }
}

/// Whether every value of `t` is finite — its sum is, as a NaN or an
/// infinity carries through every addition.
fn all_finite(t: &Tensor) -> Result<bool, SwarmError> {
    t.sum_all()
        .and_then(|s| s.to_dtype(DType::F32))
        .and_then(|s| s.to_scalar::<f32>())
        .map(f32::is_finite)
        .map_err(SwarmError::internal)
}

/// A model's earlier conversations, fitted: kernel ridge over their positions,
/// `φ·Hᵀ·A + base`, on the device the guesses run on ([`StandInPrior`]).
#[derive(Clone)]
pub struct PriorFit {
    base: Tensor,
    features: Tensor,
    weights: Tensor,
}

impl PriorFit {
    /// The positions it was fitted from.
    pub fn rows(&self) -> usize {
        self.features.dim(0).unwrap_or(0)
    }

    /// Its estimate of the run's effect at the normalised starts `phi`
    /// (`[n, hidden]`).
    fn estimate(&self, phi: &Tensor) -> Result<Tensor, SwarmError> {
        phi.matmul(&self.features.t().map_err(SwarmError::internal)?)
            .and_then(|k| k.matmul(&self.weights))
            .and_then(|e| e.broadcast_add(&self.base))
            .map_err(SwarmError::internal)
    }
}

/// What a model's skipped run did in the conversations it served before this
/// one (module docs): the last [`PRIOR_ROWS`] positions every reply's
/// stand-in learned ([`RunStandIn::collecting`]), on the host, and the fit
/// made of them once a reply is out ([`Self::absorb`]). One per loaded model
/// (`SplitModel::stand_in_prior`): a model loaded again, or placed again,
/// starts empty.
pub struct StandInPrior {
    run: (usize, usize),
    most: usize,
    lambda: f64,
    /// `(φ, Δ)`, `[≤ most, hidden]` f32 on the host, oldest first.
    rows: Option<(Tensor, Tensor)>,
    fit: Option<PriorFit>,
    /// Fits made, and the milliseconds they cost.
    pub fits: u64,
    pub ms: f64,
}

impl StandInPrior {
    /// An empty prior for the run `stand_in` stands in for.
    pub fn for_stand_in(stand_in: &RunStandIn) -> Self {
        Self {
            run: stand_in.run(),
            most: PRIOR_ROWS,
            lambda: stand_in.lambda,
            rows: None,
            fit: None,
            fits: 0,
            ms: 0.0,
        }
    }

    /// The run it was learned for.
    pub fn run(&self) -> (usize, usize) {
        self.run
    }

    /// Positions held.
    pub fn rows(&self) -> usize {
        self.rows
            .as_ref()
            .and_then(|(p, _)| p.dim(0).ok())
            .unwrap_or(0)
    }

    /// The fit for a reply whose prompt brought `phi` and `delta`
    /// ([`RunStandIn::rows_of`]): where the prompt holds more positions than
    /// the window (`window`), fitted again with them after the held ones —
    /// for this reply only, never kept; otherwise the last fit made.
    pub fn fit_for_prompt(
        &mut self,
        phi: &Tensor,
        delta: &Tensor,
        window: usize,
    ) -> Result<Option<PriorFit>, SwarmError> {
        let device = phi.device().clone();
        if phi.dim(0).map_err(SwarmError::internal)? <= window
            || !(all_finite(phi)? && all_finite(delta)?)
        {
            // Rows taken in since the last fit are fitted now, on the reply's
            // first step, where the worker had no idle moment for them.
            self.refit_while(&device, &|| true)?;
            return Ok(self.fit.clone());
        }
        let host = |t: &Tensor| t.to_device(&Device::Cpu).map_err(SwarmError::internal);
        let (all_phi, all_delta) = self.joined(&[(host(phi)?, host(delta)?)])?;
        // The kept fit stays on the host side of the card while this one is
        // in use: both at once is twice the memory for nothing.
        self.fit = None;
        self.timed_fit(&all_phi, &all_delta, &device, &|| true)?
            .ok_or_else(|| SwarmError::Internal("stand-in prior: the fit stopped".into()))
            .map(Some)
    }

    /// [`Self::take_in`], then fit again on `device` at once.
    pub fn absorb(&mut self, stand_in: &mut RunStandIn, device: &Device) -> Result<(), SwarmError> {
        self.take_in(stand_in)?;
        self.refit_while(device, &|| true).map(|_| ())
    }

    /// Take in every position `stand_in` learned (its reply's prompt and the
    /// positions the reply committed) after the held ones and keep the last
    /// [`PRIOR_ROWS`] — without fitting them: the kept fit is dropped and the
    /// prior is [`Self::stale`] until [`Self::refit_while`] (the worker, while
    /// nothing waits for it) or [`Self::fit_for_prompt`] (the next reply's
    /// first step) fits it. Fitted as the reply ended, 2,048 rows cost the
    /// next request's first token 175-280 ms on an 8 GB card whenever it came
    /// straight after (round 15). A stand-in for another run starts the prior
    /// again.
    pub fn take_in(&mut self, stand_in: &mut RunStandIn) -> Result<(), SwarmError> {
        if stand_in.run() != self.run {
            *self = Self::for_stand_in(stand_in);
        }
        // The reply's own fit goes first: the new one is made beside nothing.
        stand_in.prior = None;
        stand_in.bridge = None;
        let taken: Vec<(Tensor, Tensor)> = stand_in.taken.drain(..).collect();
        stand_in.taken_rows = 0;
        if taken.is_empty() {
            return Ok(());
        }
        self.rows = Some(self.joined(&taken)?);
        self.fit = None;
        Ok(())
    }

    /// Whether it holds rows its fit does not cover yet.
    pub fn stale(&self) -> bool {
        self.fit.is_none() && self.rows.is_some()
    }

    /// Fit the held rows on `device` where they are [`Self::stale`], asking
    /// `keep_going` between the inverse's steps (each tens of milliseconds at
    /// [`PRIOR_ROWS`]): `Ok(true)` once fitted, `Ok(false)` where there was
    /// nothing to fit or `keep_going` said stop — the rows then stay stale and
    /// nothing of the stopped fit is kept.
    pub fn refit_while(
        &mut self,
        device: &Device,
        keep_going: &dyn Fn() -> bool,
    ) -> Result<bool, SwarmError> {
        let Some((phi, delta)) = self.rows.clone().filter(|_| self.fit.is_none()) else {
            return Ok(false);
        };
        let fit = self.timed_fit(&phi, &delta, device, keep_going)?;
        let fitted = fit.is_some();
        self.fit = fit;
        Ok(fitted)
    }

    /// The held rows and `more` after them, the last `most`.
    fn joined(&self, more: &[(Tensor, Tensor)]) -> Result<(Tensor, Tensor), SwarmError> {
        let mut phis: Vec<&Tensor> = Vec::new();
        let mut deltas: Vec<&Tensor> = Vec::new();
        if let Some((p, d)) = &self.rows {
            phis.push(p);
            deltas.push(d);
        }
        for (p, d) in more {
            phis.push(p);
            deltas.push(d);
        }
        (|| -> candle_core::Result<(Tensor, Tensor)> {
            let (p, d) = (Tensor::cat(&phis, 0)?, Tensor::cat(&deltas, 0)?);
            let n = p.dim(0)?;
            let from = n.saturating_sub(self.most);
            Ok((
                p.narrow(0, from, n - from)?.contiguous()?,
                d.narrow(0, from, n - from)?.contiguous()?,
            ))
        })()
        .map_err(SwarmError::internal)
    }

    fn timed_fit(
        &mut self,
        phi: &Tensor,
        delta: &Tensor,
        device: &Device,
        keep_going: &dyn Fn() -> bool,
    ) -> Result<Option<PriorFit>, SwarmError> {
        let started = std::time::Instant::now();
        let fit = fit_rows(phi, delta, self.lambda, device, keep_going);
        if matches!(fit, Ok(Some(_))) {
            self.fits += 1;
            self.ms += started.elapsed().as_secs_f64() * 1000.0;
        }
        fit
    }
}

impl super::SplitModel {
    /// This model's prior for `stand_in`'s run — made empty where it has
    /// none, or one for another run.
    pub fn stand_in_prior_for(&mut self, stand_in: &RunStandIn) -> &mut StandInPrior {
        if self
            .stand_in_prior
            .as_ref()
            .is_none_or(|p| p.run() != stand_in.run())
        {
            self.stand_in_prior = Some(StandInPrior::for_stand_in(stand_in));
        }
        self.stand_in_prior.as_mut().expect("set above")
    }

    /// This model's prior, where a reply has made one.
    pub fn stand_in_prior(&self) -> Option<&StandInPrior> {
        self.stand_in_prior.as_ref()
    }

    /// Fit this model's prior where replies were taken in since its last fit
    /// ([`StandInPrior::refit_while`]) — the worker's use for a moment with
    /// nothing to do. `Ok(true)` once fitted; a prior that fails is dropped,
    /// costing the guesses, never the model.
    pub fn refit_stale_prior(&mut self, keep_going: &dyn Fn() -> bool) -> Result<bool, SwarmError> {
        let device = self.device().clone();
        let Some(prior) = self.stand_in_prior.as_mut().filter(|p| p.stale()) else {
            return Ok(false);
        };
        let refitted = prior.refit_while(&device, keep_going);
        if refitted.is_err() {
            self.stand_in_prior = None;
        }
        refitted
    }
}

/// Kernel ridge over `(φ, Δ)` rows on `device`: `A = (H·Hᵀ + λI)⁻¹ ·
/// (Δ − mean Δ)` with `H = φ`, the inverse by [`ridge_inverse`] — the dual of
/// the probe's `prior@λ` map, equal to it over the same rows. `None` where
/// `keep_going` stopped it (see [`ridge_inverse`]).
fn fit_rows(
    phi: &Tensor,
    delta: &Tensor,
    lambda: f64,
    device: &Device,
    keep_going: &dyn Fn() -> bool,
) -> Result<Option<PriorFit>, SwarmError> {
    let fit = (|| -> candle_core::Result<Option<PriorFit>> {
        let h = phi.to_device(device)?.contiguous()?;
        let d = delta.to_device(device)?;
        let base = d.mean(0)?;
        let left = d.broadcast_sub(&base)?.contiguous()?;
        let Some(x) = ridge_inverse(&h.matmul(&h.t()?)?, lambda, keep_going)? else {
            return Ok(None);
        };
        Ok(Some(PriorFit {
            weights: x.matmul(&left)?,
            base,
            features: h,
        }))
    })()
    .map_err(SwarmError::internal)?;
    match fit {
        Some(fit) if !(all_finite(&fit.weights)? && all_finite(&fit.base)?) => Err(
            SwarmError::Internal("stand-in prior: the fit is not finite".into()),
        ),
        fit => Ok(fit),
    }
}

/// `(K + λI)⁻¹` for a symmetric positive semi-definite `K` (`m × m`, f32) by
/// Newton–Schulz on its own device, `X ← X·(2I − A·X)` — matrix products
/// only, so a card inverts a 2,048-row Gram matrix in tens of milliseconds
/// where [`WindowInverse::from_gram`] on the host takes seconds. From
/// `X₀ = I / ‖A‖∞` (no smaller than A's largest eigenvalue, so `A·X₀`'s are
/// in (0, 1]) every `X` is a polynomial in `A`, and the residual's
/// eigenvalues square each step: slow while the smallest is near 1 — about
/// `log₂(‖A‖∞ / λ)` steps, the ridge bounding it — then quadratic down to
/// f32's floor (Pan and Schreiber 1991; arXiv 2212.02406 on its GEMM-only
/// form for GPUs). The best `X` seen is returned once `A·X` stops improving;
/// an error where it never came within [`RIDGE_INVERSE_GOOD_ENOUGH`].
/// `keep_going` is asked before every step (each waits for the card, through
/// its residual's read-back): `Ok(None)` the first time it says no.
pub(crate) fn ridge_inverse(
    k: &Tensor,
    lambda: f64,
    keep_going: &dyn Fn() -> bool,
) -> candle_core::Result<Option<Tensor>> {
    let m = k.dim(0)?;
    let eye = Tensor::eye(m, DType::F32, k.device())?;
    let a = (k.to_dtype(DType::F32)? + (&eye * lambda)?)?;
    let norm = a.abs()?.sum(1)?.max(0)?.to_scalar::<f32>()?;
    if !(norm.is_finite() && norm > 0.0) {
        candle_core::bail!("ridge inverse: the matrix's norm is {norm}");
    }
    let mut x = (&eye / f64::from(norm))?;
    let mut best: Option<(f32, Tensor)> = None;
    for _ in 0..RIDGE_INVERSE_MOST_STEPS {
        if !keep_going() {
            return Ok(None);
        }
        let t = a.matmul(&x)?;
        let off = (&t - &eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
        if !off.is_finite() {
            break;
        }
        let best_off = best.as_ref().map_or(f32::INFINITY, |(b, _)| *b);
        // Past the slow phase each step squares the residual; one that does
        // not even halve it has reached the rounding floor.
        if best_off < 0.1 && off > 0.5 * best_off {
            break;
        }
        if off < best_off {
            best = Some((off, x.clone()));
        }
        x = ((&x * 2.0)? - x.matmul(&t)?)?;
    }
    match best {
        Some((off, x)) if off <= RIDGE_INVERSE_GOOD_ENOUGH => Ok(Some(x)),
        Some((off, _)) => candle_core::bail!("ridge inverse: A·X is {off} from I"),
        None => candle_core::bail!("ridge inverse: no step was finite"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit rows, deterministic: a small LCG, so the test needs no RNG crate.
    fn unit_rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut x = seed;
        let mut next = || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        (0..n)
            .map(|_| {
                let v: Vec<f64> = (0..d).map(|_| next()).collect();
                let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                v.into_iter().map(|a| a / norm).collect()
            })
            .collect()
    }

    fn gram(rows: &[Vec<f64>]) -> Vec<f64> {
        let m = rows.len();
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                g[i * m + j] = rows[i].iter().zip(&rows[j]).map(|(a, b)| a * b).sum();
            }
        }
        g
    }

    fn largest_gap(a: &WindowInverse, b: &WindowInverse) -> f64 {
        assert_eq!(a.len(), b.len());
        a.a.iter()
            .zip(&b.a)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn the_cholesky_inverse_inverts() {
        let rows = unit_rows(12, 6, 7);
        let g = gram(&rows);
        let inv = WindowInverse::from_gram(&g, 12, 0.3).expect("positive definite");
        for i in 0..12 {
            for j in 0..12 {
                let s: f64 = (0..12)
                    .map(|k| (g[i * 12 + k] + if i == k { 0.3 } else { 0.0 }) * inv.a[k * 12 + j])
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-9, "({i},{j}) = {s}");
            }
        }
    }

    /// A window slid a row at a time — the oldest out, the newest in — is the
    /// inverse made whole from the rows it holds then.
    #[test]
    fn a_window_slid_a_row_at_a_time_is_the_window_inverted_whole() {
        let (window, d) = (10, 16);
        let rows = unit_rows(60, d, 11);
        let mut slid =
            WindowInverse::from_gram(&gram(&rows[..window]), window, 0.3).expect("first window");
        for end in window + 1..=rows.len() {
            assert!(slid.pop_front());
            let held = &rows[end - window..end - 1];
            let new = &rows[end - 1];
            let k: Vec<f64> = held
                .iter()
                .map(|r| r.iter().zip(new).map(|(a, b)| a * b).sum())
                .collect();
            assert!(slid.push(&k, new.iter().map(|a| a * a).sum()));
            let whole = WindowInverse::from_gram(&gram(&rows[end - window..end]), window, 0.3)
                .expect("window");
            let gap = largest_gap(&slid, &whole);
            assert!(gap < 1e-9, "window ending at {end}: off by {gap}");
        }
    }

    /// The pivot guard: a row pushed with a wrong self-product (here far
    /// below its true 1) is refused, never taken in.
    #[test]
    fn an_update_whose_pivot_is_impossible_is_refused() {
        let rows = unit_rows(6, 4, 3);
        let mut inv = WindowInverse::from_gram(&gram(&rows[..5]), 5, 0.3).expect("window");
        let k: Vec<f64> = rows[..5]
            .iter()
            .map(|r| r.iter().zip(&rows[5]).map(|(a, b)| a * b).sum())
            .collect();
        assert!(!inv.push(&k, -10.0));
        assert_eq!(inv.len(), 5, "a refused row is not taken in");
    }

    /// The stand-in's bridge is the probe's `recall` fit of the same window —
    /// whether its rows arrived one at a time or all at once.
    #[test]
    fn the_stand_in_fits_what_the_probe_fitted() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let (d, n) = (8usize, 40usize);
        // Skip layers 2..5 of 7: capture 1 (before) and 4 (the run's last).
        let skips = [false, false, true, true, true, false, false];
        let mut stepped = RunStandIn::for_run(&skips, &dev).expect("one run");
        let mut whole = RunStandIn::for_run(&skips, &dev).expect("one run");
        stepped.window = 16;
        whole.window = 16;
        assert_eq!(
            stepped.capture_layers(),
            &[1usize, 4].into_iter().collect::<HashSet<_>>()
        );
        let h_in: Vec<f32> = unit_rows(n, d, 5)
            .into_iter()
            .flatten()
            .map(|v| (v * 3.0) as f32)
            .collect();
        let effect: Vec<f32> = unit_rows(n, d, 9)
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect();
        let h_in = Tensor::from_vec(h_in, (1, n, d), &dev)?;
        let h_out = (&h_in + Tensor::from_vec(effect, (1, n, d), &dev)?)?;
        // Rows one or three at a time, as steps and checks bring them.
        let mut at = 0;
        let mut size = 1;
        while at < n {
            let r = size.min(n - at);
            let caught: HashMap<usize, Tensor> = [
                (1usize, h_in.narrow(1, at, r)?),
                (4usize, h_out.narrow(1, at, r)?),
            ]
            .into_iter()
            .collect();
            stepped.learn(&caught, r)?;
            at += r;
            size = if size == 1 { 3 } else { 1 };
        }
        // One capture of everything, as a prompt brings it.
        let caught: HashMap<usize, Tensor> = [(1usize, h_in.clone()), (4usize, h_out.clone())]
            .into_iter()
            .collect();
        whole.learn(&caught, n)?;
        assert!(stepped.rebuilds >= 1 && whole.rebuilds == 1);
        assert_eq!((stepped.rows(), whole.rows()), (16, 16));

        // The probe's arithmetic over the last 16 rows.
        let hi = h_in.squeeze(0)?;
        let delta = (h_out.squeeze(0)? - &hi)?;
        let phi = hi.broadcast_div(&hi.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        let f = phi.narrow(0, n - 16, 16)?;
        let base = delta.mean(0)?;
        let r = delta.narrow(0, n - 16, 16)?.broadcast_sub(&base)?;
        let g: Vec<f64> = f
            .matmul(&f.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();
        let inv = WindowInverse::from_gram(&g, 16, STAND_IN_LAMBDA).expect("window");
        let w = Tensor::from_vec(inv.as_f32(), (16, 16), &dev)?.matmul(&r)?;
        let probe = |x: &Tensor| -> candle_core::Result<Tensor> {
            x.matmul(&f.t()?)?.matmul(&w)?.broadcast_add(&base)
        };
        let queries = Tensor::from_vec(
            unit_rows(5, d, 21)
                .into_iter()
                .flatten()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
            (5, d),
            &dev,
        )?;
        let want = probe(&queries)?;
        for model in [&mut stepped, &mut whole] {
            model.refresh()?;
            let Some(SkipBridge::Recall {
                run_end,
                base,
                features,
                weights,
                prior: None,
                ..
            }) = model.bridge()
            else {
                panic!("a recall bridge");
            };
            assert_eq!(*run_end, 5);
            let got = queries
                .matmul(&features.t()?)?
                .matmul(weights)?
                .broadcast_add(base)?;
            let gap = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(gap < 1e-4, "off the probe's fit by {gap}");
        }
        Ok(())
    }

    #[test]
    fn only_one_inner_run_has_a_stand_in() {
        let dev = Device::Cpu;
        assert!(
            RunStandIn::for_run(&[true, true, false], &dev).is_none(),
            "from layer 0"
        );
        assert!(
            RunStandIn::for_run(&[false, true, true], &dev).is_none(),
            "to the end"
        );
        assert!(
            RunStandIn::for_run(&[false, true, false, true, false], &dev).is_none(),
            "two runs"
        );
        assert!(
            RunStandIn::for_run(&[false, false, false], &dev).is_none(),
            "no run"
        );
        assert!(RunStandIn::for_run(&[false, true, true, false], &dev).is_some());
    }

    type Rows = Result<(Tensor, Tensor), Box<dyn std::error::Error>>;

    /// Skip layers 2..5 of 7: the stand-in captures 1 (before) and 4 (the
    /// run's last).
    const SKIPS: [bool; 7] = [false, false, true, true, true, false, false];

    /// Unit rows leaning on one shared direction, as hidden states do: the
    /// Gram matrix is far from diagonal and, past `d` rows, singular — the
    /// ridge alone keeps it invertible.
    fn leaning_rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        unit_rows(n, d, seed)
            .into_iter()
            .map(|r| {
                let v: Vec<f64> = r
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x + if i == 0 { 1.5 } else { 0.0 })
                    .collect();
                let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                v.into_iter().map(|a| a / norm).collect()
            })
            .collect()
    }

    fn tensor(rows: &[Vec<f64>]) -> candle_core::Result<Tensor> {
        let (n, d) = (rows.len(), rows[0].len());
        Tensor::from_vec(
            rows.iter().flatten().map(|&v| v as f32).collect::<Vec<_>>(),
            (n, d),
            &Device::Cpu,
        )
    }

    /// `n` positions as a capture brings them, `[1, n, d]` each: the hidden
    /// state before the run (leaning rows, scaled) and after it, the run
    /// adding `effect(φ)`.
    fn positions(
        n: usize,
        d: usize,
        seed: u64,
        effect: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    ) -> Rows {
        let phi = tensor(&leaning_rows(n, d, seed))?;
        let h_in = (&phi * 3.0)?;
        let h_out = (&h_in + effect(&phi)?)?;
        Ok((h_in.unsqueeze(0)?, h_out.unsqueeze(0)?))
    }

    fn caught(h_in: &Tensor, h_out: &Tensor) -> HashMap<usize, Tensor> {
        [(1usize, h_in.clone()), (4usize, h_out.clone())]
            .into_iter()
            .collect()
    }

    /// `(φ, Δ)` of a capture, `[n, d]`, by the probe's arithmetic.
    fn phi_delta(h_in: &Tensor, h_out: &Tensor) -> Rows {
        let hi = h_in.squeeze(0)?;
        let delta = (h_out.squeeze(0)? - &hi)?;
        let phi = hi.broadcast_div(&hi.sqr()?.sum_keepdim(1)?.sqrt()?)?;
        Ok((phi, delta))
    }

    /// `(F·Fᵀ + λI)⁻¹` in f64 by Cholesky, as f32.
    fn exact_inverse(f: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let m = f.dim(0)?;
        let g: Vec<f64> = f
            .matmul(&f.t()?)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect();
        let inv = WindowInverse::from_gram(&g, m, STAND_IN_LAMBDA).expect("positive definite");
        Ok(Tensor::from_vec(inv.as_f32(), (m, m), &Device::Cpu)?)
    }

    /// What a bridge adds at the normalised starts `x`.
    fn bridged(bridge: &SkipBridge, x: &Tensor) -> candle_core::Result<Tensor> {
        let SkipBridge::Recall {
            base,
            features,
            weights,
            prior,
            ..
        } = bridge
        else {
            panic!("a recall bridge");
        };
        let mut e = x.matmul(&features.t()?)?.matmul(weights)?;
        if let Some((h, a)) = prior {
            e = (e + x.matmul(&h.t()?)?.matmul(a)?)?;
        }
        e.broadcast_add(base)
    }

    fn largest(t: &Tensor) -> candle_core::Result<f32> {
        t.abs()?.max_all()?.to_scalar::<f32>()
    }

    /// A model's stand-in after one reply of `n` positions, absorbed.
    fn prior_after(
        n: usize,
        d: usize,
        effect: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    ) -> Result<(StandInPrior, Tensor, Tensor), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let (h_in, h_out) = positions(n, d, 31, effect)?;
        let mut reply = RunStandIn::for_run(&SKIPS, &dev)
            .expect("one run")
            .collecting();
        reply.learn(&caught(&h_in, &h_out), n)?;
        let mut prior = StandInPrior::for_stand_in(&reply);
        prior.absorb(&mut reply, &dev)?;
        Ok((prior, h_in, h_out))
    }

    /// Newton–Schulz on the device lands where Cholesky in f64 does, on a
    /// Gram matrix shaped like a run's: 256 leaning rows in 32 dimensions,
    /// singular but for the ridge.
    #[test]
    fn the_card_inverse_is_the_cholesky_inverse() -> Result<(), Box<dyn std::error::Error>> {
        let f = tensor(&leaning_rows(256, 32, 13))?;
        let want = exact_inverse(&f)?;
        let got = ridge_inverse(&f.matmul(&f.t()?)?, STAND_IN_LAMBDA, &|| true)?.expect("fitted");
        let (gap, scale) = (largest(&(got - &want)?)?, largest(&want)?);
        assert!(gap < 1e-3 * scale, "off by {gap} of {scale}");
        Ok(())
    }

    /// A check's kept positions carry `share` of the mean of the bridge's
    /// misses at the last two of them — measured before they are learned —
    /// to the next bridge: fitted alike, it differs from one without by
    /// exactly that, everywhere.
    #[test]
    fn a_check_carries_a_share_of_its_last_miss() -> Result<(), Box<dyn std::error::Error>> {
        let (dev, d) = (Device::Cpu, 8usize);
        let (h_in, h_out) = positions(40, d, 71, |x| x.cos())?;
        let (c_in, c_out) = positions(3, d, 73, |x| (x * 2.0)?.sin())?;
        let mut runs = Vec::new();
        for share in [0.0, 0.5] {
            let mut s = RunStandIn::for_run(&SKIPS, &dev)
                .expect("one run")
                .with_carry(share, 2);
            s.learn(&caught(&h_in, &h_out), 40)?;
            s.refresh()?;
            let before = s.bridge().expect("a bridge").clone();
            s.learn(&caught(&c_in, &c_out), 3)?;
            s.refresh()?;
            runs.push((s, before));
        }
        let (phi, delta) = phi_delta(&c_in, &c_out)?;
        let last = |t: &Tensor| t.narrow(0, 1, 2);
        let at_last = runs[1].1.effect(&last(&phi)?)?.expect("a recall bridge");
        let miss = (last(&delta)? - at_last)?.mean_keepdim(0)?;
        assert!(largest(&miss)? > 0.05, "a check the bridge did not foresee");
        let queries = tensor(&leaning_rows(5, d, 79))?;
        let effect = |s: &RunStandIn| -> candle_core::Result<Tensor> {
            Ok(s.bridge()
                .expect("a bridge")
                .effect(&queries)?
                .expect("recall"))
        };
        let added = (effect(&runs[1].0)? - effect(&runs[0].0)?)?;
        let gap = largest(&added.broadcast_sub(&(miss * 0.5)?)?)?;
        assert!(gap < 1e-4, "carried {gap} off half the miss");
        Ok(())
    }

    /// A chain's far run, handed the hidden states its two ends hold, fits
    /// the bridge a run inside one model fits from the same positions
    /// captured — whichever rank each end hands over — and the bridge carries
    /// a learned start most of the way to where the far layers took it.
    #[test]
    fn a_far_run_learns_from_pairs_what_a_run_learns_from_captures(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let (h_in, h_out) = positions(40, 8, 83, |x| x.cos())?;
        let mut inside = RunStandIn::for_run(&SKIPS, &dev).expect("one run");
        inside.learn(&caught(&h_in, &h_out), 40)?;
        inside.refresh()?;
        let (start, end) = inside.run();
        let mut far = RunStandIn::for_far_run(start, end, &dev).expect("a far run");
        far.learn_pairs(&h_in.squeeze(0)?, &h_out)?;
        far.refresh()?;
        let starts = (tensor(&leaning_rows(5, 8, 89))? * 3.0)?.unsqueeze(0)?;
        let (a, b) = (inside.bridge(), far.bridge());
        let (a, b) = (a.expect("a bridge"), b.expect("a bridge"));
        let gap = largest(&(a.apply(&starts)? - b.apply(&starts)?)?)?;
        assert!(gap < 1e-5, "the two fits part by {gap}");
        let left = largest(&(b.apply(&h_in)? - &h_out)?)?;
        let whole = largest(&(&h_out - &h_in)?)?;
        assert!(left < 0.25 * whole, "{left} of {whole} still to go");
        Ok(())
    }

    /// A far run needs a first segment before it and layers in it; paired
    /// rows of two shapes are refused, and no rows teach nothing.
    #[test]
    fn a_far_run_refuses_what_it_cannot_pair() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        assert!(
            RunStandIn::for_far_run(0, 4, &dev).is_none(),
            "no first segment"
        );
        assert!(
            RunStandIn::for_far_run(4, 4, &dev).is_none(),
            "no far layers"
        );
        let mut far = RunStandIn::for_far_run(2, 5, &dev).expect("a far run");
        let (h_in, h_out) = positions(6, 8, 97, |x| x.sin())?;
        assert!(far.learn_pairs(&h_in, &h_out.narrow(1, 0, 5)?).is_err());
        far.learn_pairs(&h_in.narrow(1, 0, 0)?, &h_out.narrow(1, 0, 0)?)?;
        far.refresh()?;
        assert!(far.bridge().is_none(), "nothing learned, nothing to bridge");
        Ok(())
    }

    /// A reply's rows are taken in without a fit, and a refit stops the
    /// first time it is told to — between the inverse's steps, keeping
    /// nothing — so a request that arrives meanwhile waits one step, not the
    /// fit. The next reply's first step fits what is left.
    #[test]
    fn a_refit_stops_when_a_request_arrives() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let (mut prior, _, _) = prior_after(40, 8, |x| x.cos())?;
        assert!(!prior.stale(), "absorbed and fitted");
        let (h_in, h_out) = positions(12, 8, 43, |x| x.sin())?;
        let mut reply = RunStandIn::for_run(&SKIPS, &dev)
            .expect("one run")
            .collecting();
        reply.learn(&caught(&h_in, &h_out), 12)?;
        prior.take_in(&mut reply)?;
        assert!(prior.stale(), "taken in, not fitted");
        assert_eq!(prior.rows(), 52);

        // Asked before every step: the third ask says stop.
        let asked = std::cell::Cell::new(0);
        let until_the_third = || {
            asked.set(asked.get() + 1);
            asked.get() < 3
        };
        assert!(!prior.refit_while(&dev, &until_the_third)?);
        assert_eq!(asked.get(), 3, "stopped at the first no");
        assert!(prior.stale() && prior.fit.is_none(), "nothing of it kept");
        let fits = prior.fits;

        // A short prompt on the next reply's first step fits what was left.
        let (p_in, p_out) = positions(4, 8, 47, |x| x.sin())?;
        let (phi, delta) = phi_delta(&p_in, &p_out)?;
        let got = prior.fit_for_prompt(&phi, &delta, 16)?.expect("fitted");
        assert!(!prior.stale());
        assert_eq!(prior.fits, fits + 1);
        assert_eq!(got.rows(), 52);
        Ok(())
    }

    /// With a prior, a bridge is the probe's `prior<N>+recall` arithmetic:
    /// kernel ridge over the earlier positions, then the window fitted to
    /// what that leaves of each effect, its mean over every position seen.
    #[test]
    fn a_prior_and_the_window_add_as_the_probe_adds_them() -> Result<(), Box<dyn std::error::Error>>
    {
        let (d, earlier, n) = (8usize, 30usize, 20usize);
        let wobble = |x: &Tensor| (x.sqr()? * 0.5)?.sin();
        let (prior, e_in, e_out) = prior_after(earlier, d, wobble)?;
        assert_eq!(prior.rows(), earlier);
        let mut now = RunStandIn::for_run(&SKIPS, &Device::Cpu).expect("one run");
        now.window = 16;
        now.set_prior(prior.fit.clone())?;
        let (h_in, h_out) = positions(n, d, 37, |x| x.cos())?;
        now.learn(&caught(&h_in, &h_out), n)?;
        now.refresh()?;

        let (ep, ed) = phi_delta(&e_in, &e_out)?;
        let pbase = ed.mean(0)?;
        let a = exact_inverse(&ep)?.matmul(&ed.broadcast_sub(&pbase)?)?;
        let from_prior = |x: &Tensor| -> candle_core::Result<Tensor> {
            x.matmul(&ep.t()?)?.matmul(&a)?.broadcast_add(&pbase)
        };
        let (p, dl) = phi_delta(&h_in, &h_out)?;
        let left = (&dl - from_prior(&p)?)?;
        let lbase = left.mean(0)?;
        let f = p.narrow(0, n - 16, 16)?;
        let w = exact_inverse(&f)?.matmul(&left.narrow(0, n - 16, 16)?.broadcast_sub(&lbase)?)?;
        let queries = tensor(&leaning_rows(5, d, 41))?;
        let want = (from_prior(&queries)? + queries.matmul(&f.t()?)?.matmul(&w)?)?
            .broadcast_add(&lbase)?;
        let got = bridged(now.bridge().expect("a bridge"), &queries)?;
        let gap = largest(&(got - &want)?)?;
        assert!(gap < 1e-3 * largest(&want)?, "off the probe's fit by {gap}");
        Ok(())
    }

    /// What earlier replies learned carries to the next: a run that adds one
    /// linear map everywhere, seen at 200 earlier positions, is estimated at
    /// a new reply's tenth position far better than those ten can manage
    /// alone — the window cannot see a 16-wide map in 10 rows.
    #[test]
    fn a_reply_starts_from_what_the_earlier_replies_learned(
    ) -> Result<(), Box<dyn std::error::Error>> {
        let d = 16usize;
        let map = tensor(&unit_rows(d, d, 53))?;
        let linear = |x: &Tensor| x.matmul(&map);
        let (prior, _, _) = prior_after(200, d, linear)?;
        let (h_in, h_out) = positions(10, d, 59, linear)?;
        let queries = tensor(&leaning_rows(20, d, 61))?;
        let truth = queries.matmul(&map)?;
        let mut errors = Vec::new();
        for fit in [prior.fit.clone(), None] {
            let mut now = RunStandIn::for_run(&SKIPS, &Device::Cpu).expect("one run");
            now.set_prior(fit)?;
            now.learn(&caught(&h_in, &h_out), 10)?;
            now.refresh()?;
            let est = bridged(now.bridge().expect("a bridge"), &queries)?;
            errors.push(largest(&(est - &truth)?)?);
        }
        assert!(
            errors[0] < 0.5 * errors[1],
            "with the prior {}, without {}",
            errors[0],
            errors[1]
        );
        Ok(())
    }

    /// A prior keeps the most recent positions, oldest first, up to its
    /// limit — across replies.
    #[test]
    fn a_prior_keeps_the_last_positions_it_saw() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let mut prior: Option<StandInPrior> = None;
        let mut all = Vec::new();
        for seed in [3u64, 5, 7] {
            let (h_in, h_out) = positions(6, 4, seed, |x| x.cos())?;
            all.push(phi_delta(&h_in, &h_out)?.0);
            let mut reply = RunStandIn::for_run(&SKIPS, &dev)
                .expect("one run")
                .collecting();
            reply.learn(&caught(&h_in, &h_out), 6)?;
            let p = prior.get_or_insert_with(|| {
                let mut p = StandInPrior::for_stand_in(&reply);
                p.most = 10;
                p
            });
            p.absorb(&mut reply, &dev)?;
            assert!(reply.taken.is_empty(), "absorbed, not copied");
        }
        let prior = prior.expect("made");
        assert_eq!(prior.rows(), 10);
        let want = Tensor::cat(&all, 0)?.narrow(0, 8, 10)?;
        let (held, _) = prior.rows.as_ref().expect("rows");
        assert_eq!(largest(&(held - want)?)?, 0.0);
        Ok(())
    }

    /// A prompt longer than the window joins the prior for its own reply
    /// only; a short one reads the kept fit.
    #[test]
    fn a_long_prompt_joins_the_prior_for_its_own_reply() -> Result<(), Box<dyn std::error::Error>> {
        let (mut prior, _, _) = prior_after(30, 8, |x| x.cos())?;
        let (h_in, h_out) = positions(20, 8, 67, |x| x.sin())?;
        let (phi, delta) = phi_delta(&h_in, &h_out)?;
        let short = prior
            .fit_for_prompt(&phi, &delta, 32)?
            .expect("the kept fit");
        assert_eq!(short.rows(), 30);
        let long = prior.fit_for_prompt(&phi, &delta, 16)?.expect("a fit");
        assert_eq!(long.rows(), 50, "the held positions, then the prompt's");
        assert_eq!(
            prior.rows(),
            30,
            "the prompt is not kept until the reply is out"
        );
        assert!(prior.fit.is_none(), "the kept fit is not held beside it");
        Ok(())
    }

    /// A forward whose hidden state is not finite teaches nothing: the
    /// window keeps its inverse and the prior its fit, where one NaN row would
    /// have cost both for thousands of positions.
    #[test]
    fn a_position_that_is_not_finite_is_left_out() -> Result<(), Box<dyn std::error::Error>> {
        let dev = Device::Cpu;
        let (h_in, h_out) = positions(20, 4, 71, |x| x.cos())?;
        let mut reply = RunStandIn::for_run(&SKIPS, &dev)
            .expect("one run")
            .collecting();
        reply.learn(&caught(&h_in, &h_out), 20)?;
        let bad_out = (h_out.narrow(1, 0, 1)? * f64::NAN)?;
        reply.learn(&caught(&h_in.narrow(1, 0, 1)?, &bad_out), 1)?;
        assert_eq!((reply.not_finite, reply.rows()), (1, 20));
        reply.refresh()?;
        let est = bridged(
            reply.bridge().expect("still a bridge"),
            &tensor(&leaning_rows(3, 4, 73))?,
        )?;
        assert!(all_finite(&est)?);
        let mut prior = StandInPrior::for_stand_in(&reply);
        prior.absorb(&mut reply, &dev)?;
        assert_eq!(prior.rows(), 20, "the refused position is not kept");
        assert!(prior.fit.is_some());
        Ok(())
    }

    #[test]
    fn a_prior_is_set_before_anything_is_learned() -> Result<(), Box<dyn std::error::Error>> {
        let (prior, h_in, h_out) = prior_after(12, 4, |x| x.cos())?;
        let mut now = RunStandIn::for_run(&SKIPS, &Device::Cpu).expect("one run");
        now.learn(&caught(&h_in, &h_out), 12)?;
        assert!(now.set_prior(prior.fit.clone()).is_err());
        Ok(())
    }
}
