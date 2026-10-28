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
    /// `Σ Δ` over every position seen, and their count: the bridge's base is
    /// their mean, as the probe's was.
    delta_sum: Option<Tensor>,
    seen: usize,
    inverse: Option<WindowInverse>,
    /// Rows added since the inverse was last rebuilt whole.
    since_rebuild: usize,
    bridge: Option<SkipBridge>,
    /// Inverses rebuilt whole, and the milliseconds the stand-in has cost.
    pub rebuilds: u64,
    pub ms: f64,
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
        Some(Self {
            run_start,
            run_end,
            capture: [run_start - 1, run_end - 1].into_iter().collect(),
            device: device.clone(),
            window: STAND_IN_WINDOW,
            lambda: STAND_IN_LAMBDA,
            phi: None,
            delta: None,
            delta_sum: None,
            seen: 0,
            inverse: None,
            since_rebuild: 0,
            bridge: None,
            rebuilds: 0,
            ms: 0.0,
        })
    }

    /// The same, fitted from the last `window` positions instead of
    /// [`STAND_IN_WINDOW`] (the probe's `PROBE_WINDOW`).
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(LEAST_ROWS);
        self
    }

    /// The layers a forward captures for [`Self::learn`]: the one before the
    /// run and the run's last.
    pub fn capture_layers(&self) -> &HashSet<usize> {
        &self.capture
    }

    /// The most positions it is fitted from: what a prompt pass captures
    /// for it (`split::Capture::last`).
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
        let started = std::time::Instant::now();
        let out = self.learn_inner(caught, kept);
        self.ms += started.elapsed().as_secs_f64() * 1000.0;
        out
    }

    fn learn_inner(
        &mut self,
        caught: &HashMap<usize, Tensor>,
        kept: usize,
    ) -> Result<(), SwarmError> {
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
        let rows = (|| -> candle_core::Result<(Tensor, Tensor, Tensor)> {
            let delta = (&h_out - &h_in)?;
            let norm = h_in.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-6, f64::MAX)?;
            let phi = h_in.broadcast_div(&norm)?;
            let sum = delta.sum(0)?;
            Ok((phi, delta, sum))
        })()
        .map_err(SwarmError::internal)?;
        let (x_phi, x_delta, x_sum) = rows;
        self.delta_sum = Some(match self.delta_sum.take() {
            Some(s) => (s + x_sum).map_err(SwarmError::internal)?,
            None => x_sum,
        });
        self.seen += kept;

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
        let (Some(inverse), Some(phi), Some(delta), Some(sum)) =
            (&self.inverse, &self.phi, &self.delta, &self.delta_sum)
        else {
            return Ok(None);
        };
        let m = inverse.len();
        if m < LEAST_ROWS || self.seen == 0 {
            return Ok(None);
        }
        (|| -> candle_core::Result<Option<SkipBridge>> {
            let base = (sum / self.seen as f64)?;
            let inv = Tensor::from_vec(inverse.as_f32(), (m, m), &self.device)?;
            let weights = inv.matmul(&delta.broadcast_sub(&base)?.contiguous()?)?;
            Ok(Some(SkipBridge::Recall {
                run_end: self.run_end,
                base,
                features: phi.contiguous()?,
                weights,
            }))
        })()
        .map_err(SwarmError::internal)
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
}
