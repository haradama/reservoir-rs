//! Hénon map (Henon attractor) time-series generator.
//!
//! This module provides a small, dependency-light implementation of the
//! 2D discrete-time Hénon map, often used as a chaotic benchmark in
//! reservoir computing / ESN experiments.
//!
//! # Notes
//! - This implementation generates a **scalar** time series by returning `x_t`.
//! - A short **burn-in** is performed in [`HenonMap::generate`] to reduce
//!   dependence on the initial condition.
//! - Requires `alloc` because generated samples are stored in `Vec`.

extern crate alloc;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Parameters for the Hénon map.
///
/// The Hénon map is defined as:
///
/// ```text
/// x_{t+1} = 1 - a * x_t^2 + y_t
/// y_{t+1} = b * x_t
/// ```
///
/// # Fields
/// - `a`, `b`: map coefficients (commonly `a=1.4`, `b=0.3`)
/// - `x0`, `y0`: initial state
/// - `steps`: number of samples to generate (after burn-in)
#[derive(Clone, Debug)]
pub struct HenonParams {
    /// Coefficient `a` in `x_{t+1} = 1 - a*x_t^2 + y_t`.
    pub a: f64,
    /// Coefficient `b` in `y_{t+1} = b*x_t`.
    pub b: f64,
    /// Initial x state `x_0`.
    pub x0: f64,
    /// Initial y state `y_0`.
    pub y0: f64,
    /// Number of samples to generate (after burn-in).
    pub steps: usize,
}

impl Default for HenonParams {
    fn default() -> Self {
        Self {
            // Canonical chaotic parameter set.
            a: 1.4,
            b: 0.3,
            // Commonly used initial condition.
            x0: 0.0,
            y0: 0.0,
            // Reasonable default length for quick experiments.
            steps: 2000,
        }
    }
}

/// Stateful Hénon map iterator.
///
/// Holds the current `(x, y)` state and updates it on each call to [`step`](HenonMap::step).
///
/// # Example
/// ```
/// use reservoir_datasets::henon::{HenonMap, HenonParams};
///
/// let mut map = HenonMap::new(HenonParams::default());
/// let x1 = map.step();
/// let x2 = map.step();
/// println!("{x1}, {x2}");
/// ```
pub struct HenonMap {
    /// Immutable parameters of the map (coefficients, initial values, length).
    params: HenonParams,
    /// Current x state (updated each step).
    current_x: f64,
    /// Current y state (updated each step).
    current_y: f64,
}

impl HenonMap {
    /// Creates a new Hénon map generator initialized at `(x0, y0)`.
    ///
    /// # Arguments
    /// - `params`: coefficients and initial conditions
    ///
    /// # Notes
    /// The internal state is initialized from `params.x0` and `params.y0`.
    pub fn new(params: HenonParams) -> Self {
        Self {
            current_x: params.x0,
            current_y: params.y0,
            params,
        }
    }

    /// Advances the map by one step and returns the next `x` value.
    ///
    /// This updates the internal state from `(x_t, y_t)` to `(x_{t+1}, y_{t+1})`.
    ///
    /// # Returns
    /// - `x_{t+1}` (scalar time-series output)
    ///
    /// # Panics
    /// This function does not panic by itself, but floating-point overflow/NaN
    /// can occur for extreme parameter choices or initial conditions.
    pub fn step(&mut self) -> f64 {
        let x = self.current_x;
        let y = self.current_y;

        // Hénon recurrence
        let next_x = 1.0 - self.params.a * x.powi(2) + y;
        let next_y = self.params.b * x;

        self.current_x = next_x;
        self.current_y = next_y;

        next_x
    }

    /// Generates a time series of length `params.steps`.
    ///
    /// # Behavior
    /// - Runs a fixed **burn-in** (warm-up) of 100 steps to approach the attractor.
    /// - Then collects `steps` samples of `x_{t+1}` into a `Vec<f64>`.
    ///
    /// # Returns
    /// A vector containing `params.steps` samples.
    ///
    /// # Notes
    /// If you need both `x` and `y`, you can call [`step`](HenonMap::step) manually
    /// and track `current_y` yourself (or extend this API in the future).
    pub fn generate(&mut self) -> Vec<f64> {
        let mut data = Vec::with_capacity(self.params.steps);

        // Burn-in: discard transient dynamics so outputs are closer to the attractor.
        for _ in 0..100 {
            self.step();
        }

        // Collect the requested number of samples.
        for _ in 0..self.params.steps {
            data.push(self.step());
        }

        data
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-15;

    #[test]
    fn test_henon_map_step() {
        let params = HenonParams {
            a: 1.4,
            b: 0.3,
            x0: 0.0,
            y0: 0.0,
            steps: 10,
        };
        let mut map = HenonMap::new(params);

        let next_x = map.step();
        assert!((next_x - 1.0).abs() < EPSILON);
        assert!((map.current_x - 1.0).abs() < EPSILON);
        assert!((map.current_y - 0.0).abs() < EPSILON);

        let next_x_2 = map.step();
        assert!((next_x_2 - (-0.4)).abs() < EPSILON);
        assert!((map.current_x - (-0.4)).abs() < EPSILON);
        assert!((map.current_y - 0.3).abs() < EPSILON);
    }

    #[test]
    fn test_henon_map_generate_size() {
        let params = HenonParams {
            steps: 50,
            ..Default::default()
        };
        let mut map = HenonMap::new(params);
        let data = map.generate();
        assert_eq!(data.len(), 50);
    }
}
