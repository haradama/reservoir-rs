extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use rand::{Rng, SeedableRng};

use crate::RngType;

#[derive(Clone, Debug)]
pub struct NarmaParams {
    pub n: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub delta: f64,
    pub steps: usize,
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

pub struct Narma {
    params: NarmaParams,
    y_history: VecDeque<f64>,
    u_history: VecDeque<f64>,
    rng: RngType,
}

impl Narma {
    pub fn new(params: NarmaParams) -> Self {
        let n = params.n;

        let mut y_history = VecDeque::with_capacity(n);
        let mut u_history = VecDeque::with_capacity(n);
        for _ in 0..n {
            y_history.push_back(0.0);
            u_history.push_back(0.0);
        }

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

    pub fn step(&mut self) -> (f64, f64) {
        let u_t: f64 = self.rng.gen::<f64>() * 0.5;

        let y_t = *self.y_history.back().unwrap();
        let sum_y: f64 = self.y_history.iter().sum();

        let u_delayed = if self.params.n > 1 {
            self.u_history[1]
        } else {
            u_t
        };

        let p = &self.params;
        let next_y = p.alpha * y_t + p.beta * y_t * sum_y + p.gamma * u_delayed * u_t + p.delta;

        self.u_history.pop_front();
        self.u_history.push_back(u_t);

        self.y_history.pop_front();
        self.y_history.push_back(next_y);

        (u_t, next_y)
    }

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
