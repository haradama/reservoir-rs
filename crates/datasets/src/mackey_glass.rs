//! Mackey–Glass delay differential equation (DDE) time-series generator.
//!
//! This module provides a small implementation of the Mackey–Glass system,
//! commonly used as a chaotic benchmark in reservoir computing / ESN literature.
//!
//! The continuous-time equation is typically written as:
//!
//! ```text
//! dx/dt = -b * x(t) + a * x(t - tau) / (1 + x(t - tau)^n)
//! ```
//!
//! Here we simulate it with a fixed-step Runge–Kutta 4 (RK4) integrator,
//! and approximate the delay term `x(t - tau)` using a ring buffer of past
//! samples.
//!
//! # Notes
//! - Requires `alloc` because we store a history buffer and return `Vec<f64>`.
//! - The delay buffer length is `ceil(tau / h)`.
//! - History initialization can be randomized (seeded) or supplied explicitly.

extern crate alloc;

use crate::RngType;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float;
use rand::{Rng, SeedableRng};

/// Parameters for the Mackey–Glass time series.
///
/// # Fields
/// - `a`, `b`, `n`: equation parameters (common defaults include `a=0.2, b=0.1, n=10`)
/// - `tau`: delay amount (in the same time unit as `h`)
/// - `x0`: initial (current) state `x(t0)`
/// - `h`: integration time step
/// - `steps`: number of samples to generate
/// - `seed`: RNG seed used when `history` is `None`
/// - `history`: optional explicit history buffer used to define `x(t - tau)`
///
/// # Delay buffer length
/// Internally we keep `history_len = ceil(tau / h)` samples.
/// The oldest element corresponds to `x(t - tau)` (approximately).
#[derive(Clone, Debug)]
pub struct MackeyGlassParams {
    /// Coefficient `a` in the non-linear delayed feedback term.
    pub a: f64,
    /// Coefficient `b` in the linear decay term.
    pub b: f64,
    /// Exponent `n` in `1 + x(t - tau)^n`.
    pub n: u32,
    /// Delay `tau` (time units).
    pub tau: usize,
    /// Initial state `x0 = x(t0)`.
    pub x0: f64,
    /// Integration time step.
    pub h: f64,
    /// Number of samples to generate.
    pub steps: usize,
    /// RNG seed used when `history` is not provided.
    pub seed: Option<u64>,
    /// Optional explicit delay history.
    ///
    /// If provided, the last `ceil(tau/h)` values are taken as the initial delay line.
    /// **Important:** this code assumes `history.len() >= ceil(tau/h)`.
    pub history: Option<Vec<f64>>,
}

/// Stateful Mackey–Glass simulator.
///
/// This struct holds:
/// - parameters,
/// - a delay-line buffer (`history`),
/// - the current state `x(t)`.
///
/// Each call to [`step`](MackeyGlass::step) advances time by `h` and returns the
/// next state.
pub struct MackeyGlass {
    /// Immutable configuration parameters.
    params: MackeyGlassParams,
    /// Delay-line buffer of past values (oldest at the front).
    history: VecDeque<f64>,
    /// Current state `x(t)`.
    current_x: f64,
}

impl MackeyGlass {
    /// Creates a new Mackey–Glass simulator.
    ///
    /// # Delay buffer initialization
    /// The delay buffer length is:
    /// `history_len = ceil(tau / h)`.
    ///
    /// - If `params.history` is `Some`, this uses the **last** `history_len` values.
    /// - Otherwise, it fills the buffer with small random noise around `x0`.
    ///
    /// # Seed behavior
    /// - If `seed` is `Some`, it is used deterministically.
    /// - If `seed` is `None`:
    ///   - with `std`: uses entropy (`StdRng::from_entropy()`),
    ///   - without `std`: falls back to a fixed seed (`42`) for reproducibility.
    ///
    /// # Panics
    /// If `params.history` is provided but shorter than `ceil(tau/h)`,
    /// slicing will panic. (Consider validating this in a future revision.)
    pub fn new(params: MackeyGlassParams) -> Self {
        // Number of discrete steps required to represent the delay tau with step size h.
        let history_len = (params.tau as f64 / params.h).ceil() as usize;
        let mut history = VecDeque::with_capacity(history_len);

        if let Some(hist) = &params.history {
            // Use the tail of the provided history as the initial delay line.
            // Assumes hist is long enough; otherwise this slice will panic.
            history.extend(hist[hist.len() - history_len..].iter().cloned());
        } else {
            // Initialize the delay line with small noise around x0.
            let mut rng = match params.seed {
                Some(s) => RngType::seed_from_u64(s),
                None => {
                    #[cfg(feature = "std")]
                    {
                        RngType::from_entropy()
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        // Deterministic fallback for no_std builds.
                        RngType::seed_from_u64(42)
                    }
                }
            };
            for _ in 0..history_len {
                history.push_back(params.x0 + rng.gen::<f64>() * 0.1);
            }
        }

        Self {
            current_x: params.x0,
            params,
            history,
        }
    }

    /// Advances the system by one integration step (`h`) using RK4.
    ///
    /// This approximates the DDE by using `x(t - tau)` taken from the front of
    /// the delay buffer.
    ///
    /// # Returns
    /// The newly computed value `x(t + h)`.
    ///
    /// # Implementation details
    /// - The derivative function is:
    ///   `f(x, x_tau) = -b*x + a*x_tau / (1 + x_tau^n)`
    /// - RK4 is applied with fixed step size `h`.
    /// - After integration, the delay buffer is updated by popping the oldest
    ///   value and pushing the new one.
    pub fn step(&mut self) -> f64 {
        let xt = self.current_x;

        // Approximate x(t - tau) using the oldest stored sample.
        // `unwrap()` is safe as long as history_len > 0.
        let xtau = self.history.front().copied().unwrap();

        // DDE RHS: dx/dt = -b*x + a*x_tau / (1 + x_tau^n)
        let f = |x: f64, x_tau: f64, p: &MackeyGlassParams| {
            -p.b * x + p.a * x_tau / (1. + x_tau.powi(p.n as i32))
        };

        let h = self.params.h;

        // Classic RK4 integration.
        let k1 = h * f(xt, xtau, &self.params);
        let k2 = h * f(xt + 0.5 * k1, xtau, &self.params);
        let k3 = h * f(xt + 0.5 * k2, xtau, &self.params);
        let k4 = h * f(xt + k3, xtau, &self.params);

        let new_x = xt + (k1 + 2. * k2 + 2. * k3 + k4) / 6.;

        // Update delay line and current state.
        self.history.pop_front();
        self.history.push_back(new_x);
        self.current_x = new_x;

        new_x
    }

    /// Generates a time series of length `params.steps`.
    ///
    /// # Returns
    /// A vector containing successive samples of `x(t)` after each integration step.
    ///
    /// # Notes
    /// This method does not perform an explicit burn-in. If you want to discard
    /// transients, call [`step`](MackeyGlass::step) a number of times before
    /// collecting samples, or discard a prefix of the returned vector.
    pub fn generate(&mut self) -> Vec<f64> {
        (0..self.params.steps).map(|_| self.step()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_mackey_glass_generate_size() {
        let params = MackeyGlassParams {
            a: 0.2,
            b: 0.1,
            n: 10,
            tau: 17,
            x0: 1.2,
            h: 0.1,
            steps: 100,
            seed: Some(42),
            history: None,
        };
        let mut mg = MackeyGlass::new(params);
        let data = mg.generate();
        assert_eq!(data.len(), 100);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_mackey_glass_history_size() {
        let params = MackeyGlassParams {
            a: 0.2,
            b: 0.1,
            n: 10,
            tau: 17,
            x0: 1.2,
            h: 0.1,
            steps: 100,
            seed: Some(42),
            history: None,
        };
        let mg = MackeyGlass::new(params);
        assert_eq!(mg.history.len(), 170);
    }
}
