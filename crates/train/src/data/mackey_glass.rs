use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::VecDeque;

pub struct MackeyGlassParams {
    pub a: f64,
    pub b: f64,
    pub n: u32,
    pub tau: usize,
    pub x0: f64,
    pub h: f64,
    pub steps: usize,
    pub seed: Option<u64>,
    pub history: Option<Vec<f64>>,
}

pub struct MackeyGlass {
    params: MackeyGlassParams,
    history: VecDeque<f64>,
    current_x: f64,
}

impl MackeyGlass {
    pub fn new(params: MackeyGlassParams) -> Self {
        let history_len = (params.tau as f64 / params.h).ceil() as usize;
        let mut history = VecDeque::with_capacity(history_len);

        if let Some(hist) = &params.history {
            history.extend(hist[hist.len() - history_len..].iter().cloned());
        } else {
            let mut rng = match params.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
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

    pub fn step(&mut self) -> f64 {
        let xt = self.current_x;
        let xtau = self.history.front().copied().unwrap();
        let f = |x: f64, x_tau: f64, p: &MackeyGlassParams| {
            -p.b * x + p.a * x_tau / (1. + x_tau.powi(p.n as i32))
        };
        let h = self.params.h;
        let k1 = h * f(xt, xtau, &self.params);
        let k2 = h * f(xt + 0.5 * k1, xtau, &self.params);
        let k3 = h * f(xt + 0.5 * k2, xtau, &self.params);
        let k4 = h * f(xt + k3, xtau, &self.params);
        let new_x = xt + (k1 + 2. * k2 + 2. * k3 + k4) / 6.;

        self.history.pop_front();
        self.history.push_back(new_x);
        self.current_x = new_x;
        new_x
    }

    pub fn generate(&mut self) -> Vec<f64> {
        (0..self.params.steps).map(|_| self.step()).collect()
    }
}
