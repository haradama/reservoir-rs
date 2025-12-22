//! NARMA (Nonlinear AutoRegressive Moving Average) benchmark generator.
//!
//! NARMA is a standard synthetic task used in reservoir computing to evaluate
//! memory and nonlinearity. A common form is:
//!
//! ```text
//! y(t+1) = α y(t) + β y(t) * Σ_{i=0}^{n-1} y(t-i) + γ u(t) u(t-d) + δ
//! ```
//!
//! where `u(t)` is an exogenous random input signal, and the task is to predict
//! `y(t)` from `u(t)` and/or past values.
//!
//! This module produces both the input sequence `u` and target sequence `y`.
//!
//! # Notes
//! - Requires `alloc` because this implementation uses `VecDeque` and returns `Vec<f64>`.
//! - Inputs are generated as uniform random numbers in `[0, 0.5)`.
//! - The delay term `u(t-d)` is approximated using an input history buffer.
//!   (Current implementation uses `u_history[1]` as the delayed value.)

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;
use rand::{Rng, SeedableRng};

use crate::RngType;

/// Parameters for the NARMA generator.
///
/// # Fields
/// - `n`: order / memory length (history size)
/// - `alpha`, `beta`, `gamma`, `delta`: equation parameters
/// - `steps`: number of samples to generate
/// - `seed`: RNG seed for reproducible inputs
///
/// # Default
/// The provided defaults are commonly used for quick benchmarks, but different
/// papers use different parameter sets (e.g., NARMA10 is typical).
#[derive(Clone, Debug)]
pub struct NarmaParams {
    /// Order of the NARMA system (history length).
    pub n: usize,
    /// Linear autoregressive coefficient.
    pub alpha: f64,
    /// Nonlinear feedback coefficient.
    pub beta: f64,
    /// Input coupling coefficient.
    pub gamma: f64,
    /// Constant bias term.
    pub delta: f64,
    /// Number of steps (samples) to generate.
    pub steps: usize,
    /// RNG seed for generating `u(t)`. If `None`, a fixed seed is used.
    pub seed: Option<u64>,
}

impl Default for NarmaParams {
    fn default() -> Self {
        Self {
            n: 10,
            alpha: 0.3,
            beta: 0.05,
            gamma: 1.5,
            delta: 0.1,
            steps: 2000,
            seed: None,
        }
    }
}

/// Stateful NARMA generator.
///
/// Maintains history buffers for:
/// - `y_history`: past outputs `y(t-i)`
/// - `u_history`: past inputs `u(t-i)`
///
/// and an RNG used to generate the input `u(t)`.
pub struct Narma {
    /// Immutable configuration parameters.
    params: NarmaParams,
    /// Output history buffer (oldest at the front).
    y_history: VecDeque<f64>,
    /// Input history buffer (oldest at the front).
    u_history: VecDeque<f64>,
    /// RNG for generating input `u(t)`.
    rng: RngType,
}

impl Narma {
    /// Creates a new NARMA generator.
    ///
    /// Initializes `y_history` and `u_history` with `n` zeros.
    ///
    /// # Seed behavior
    /// - If `params.seed` is `Some`, it is used deterministically.
    /// - If `params.seed` is `None`, this implementation uses a fixed seed (`42`)
    ///   to keep output deterministic across runs.
    ///
    /// # Panics
    /// If `params.n == 0`, `VecDeque::with_capacity(0)` is fine, but later
    /// indexing/`back().unwrap()` in [`step`](Narma::step) will panic.
    /// (Consider validating `n >= 1` if you want a stricter public API.)
    pub fn new(params: NarmaParams) -> Self {
        let n = params.n;

        // Initialize histories with zeros so early steps are well-defined.
        let mut y_history = VecDeque::with_capacity(n);
        let mut u_history = VecDeque::with_capacity(n);
        for _ in 0..n {
            y_history.push_back(0.0);
            u_history.push_back(0.0);
        }

        // Deterministic RNG initialization.
        let rng = match params.seed {
            Some(s) => RngType::seed_from_u64(s),
            None => RngType::seed_from_u64(42),
        };

        Self {
            params,
            y_history,
            u_history,
            rng,
        }
    }

    /// Advances the system by one step and returns `(u(t), y(t+1))`.
    ///
    /// # Input generation
    /// The input `u(t)` is sampled uniformly and scaled into `[0, 0.5)`.
    ///
    /// # Target update
    /// Uses the current `y(t)` (the last element of `y_history`) and the sum of
    /// the last `n` outputs:
    ///
    /// ```text
    /// sum_y = Σ y(t-i) for i = 0..n-1
    /// ```
    ///
    /// The delayed input term is currently:
    /// - if `n > 1`: `u_delayed = u_history[1]`
    /// - else: `u_delayed = u(t)`
    ///
    /// Then:
    ///
    /// ```text
    /// y(t+1) = α y(t) + β y(t) * sum_y + γ u(t) * u_delayed + δ
    /// ```
    ///
    /// # Notes on delay indexing
    /// `u_history[1]` corresponds to a *specific* delay choice relative to how
    /// the buffer is updated (pop-front then push-back). This is fine as a simple
    /// benchmark, but if you want a canonical “NARMA-n” definition (often uses a
    /// fixed delay like 10), consider making the delay explicit in parameters.
    pub fn step(&mut self) -> (f64, f64) {
        // Random exogenous input signal.
        let u_t: f64 = self.rng.gen::<f64>() * 0.5;

        // Current output and sum of the last n outputs.
        let y_t = *self.y_history.back().unwrap();
        let sum_y: f64 = self.y_history.iter().sum();

        // Delayed input term (simple choice based on buffer contents).
        let u_delayed = if self.params.n > 1 {
            self.u_history[1]
        } else {
            u_t
        };

        let p = &self.params;
        let next_y = p.alpha * y_t + p.beta * y_t * sum_y + p.gamma * u_delayed * u_t + p.delta;

        // Update histories: drop oldest sample and append newest.
        self.u_history.pop_front();
        self.u_history.push_back(u_t);

        self.y_history.pop_front();
        self.y_history.push_back(next_y);

        (u_t, next_y)
    }

    /// Generates `(inputs, targets)` sequences of length `params.steps`.
    ///
    /// # Returns
    /// - `inputs`: `u(t)` values
    /// - `targets`: `y(t+1)` values produced by the recurrence
    ///
    /// This pairing is convenient for supervised training where the model uses
    /// `u(t)` to predict `y(t+1)` (or `y(t)` depending on how you align data).
    pub fn generate(&mut self) -> (Vec<f64>, Vec<f64>) {
        let mut inputs = Vec::with_capacity(self.params.steps);
        let mut targets = Vec::with_capacity(self.params.steps);

        for _ in 0..self.params.steps {
            let (u, y) = self.step();
            inputs.push(u);
            targets.push(y);
        }

        (inputs, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_narma_init() {
        let params = NarmaParams {
            n: 5,
            ..Default::default()
        };
        let narma = Narma::new(params);

        assert_eq!(narma.y_history.len(), 5);
        assert_eq!(narma.u_history.len(), 5);
        assert!(narma.y_history.iter().all(|&v| v == 0.0));
        assert!(narma.u_history.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_narma_generate_size() {
        let params = NarmaParams {
            steps: 75,
            ..Default::default()
        };
        let mut narma = Narma::new(params);
        let (inputs, targets) = narma.generate();

        assert_eq!(inputs.len(), 75);
        assert_eq!(targets.len(), 75);
    }

    #[test]
    fn test_narma_step_update() {
        let params = NarmaParams {
            n: 3,
            seed: Some(42),
            ..Default::default()
        };
        let mut narma = Narma::new(params);

        let (u1, y1) = narma.step();
        assert_ne!(u1, 0.0);
        assert_ne!(y1, 0.0);

        assert_eq!(narma.y_history.front().copied().unwrap(), 0.0);
        assert_eq!(narma.y_history.back().copied().unwrap(), y1);
        assert_eq!(narma.u_history.front().copied().unwrap(), 0.0);
        assert_eq!(narma.u_history.back().copied().unwrap(), u1);

        let (u2, y2) = narma.step();

        assert_eq!(narma.y_history.front().copied().unwrap(), 0.0);
        assert_eq!(narma.y_history.back().copied().unwrap(), y2);
        assert_eq!(narma.u_history.front().copied().unwrap(), 0.0);
        assert_eq!(narma.u_history.back().copied().unwrap(), u2);
    }
}
