use reservoir_dense::ESNBuilder;
use reservoir_eval::{rmse, rsquare};

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
            if hist.len() < history_len {
                panic!(
                    "Insufficient history: expected at least {} values.",
                    history_len
                );
            }
            history.extend(hist[hist.len() - history_len..].iter().cloned());
        } else {
            let mut rng = match params.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            for _ in 0..history_len {
                let noise = 0.2 * (rng.gen::<f64>() - 0.5);
                history.push_back(params.x0 + noise);
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
        let xtau = if self.params.tau == 0 {
            0.0
        } else {
            let val = self.history.pop_front().unwrap();
            self.history.push_back(xt);
            val
        };
        self.current_x = Self::rk4_step(xt, xtau, &self.params);
        self.current_x
    }

    pub fn generate(&mut self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.params.steps);
        for _ in 0..self.params.steps {
            result.push(self.step());
        }
        result
    }

    fn rk4_step(xt: f64, xtau: f64, p: &MackeyGlassParams) -> f64 {
        let f = |x: f64| -> f64 { -p.b * x + p.a * xtau / (1.0 + xtau.powi(p.n as i32)) };
        let h = p.h;
        let k1 = h * f(xt);
        let k2 = h * f(xt + 0.5 * k1);
        let k3 = h * f(xt + 0.5 * k2);
        let k4 = h * f(xt + k3);

        xt + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    }
}

fn main() {
    let mut mg = MackeyGlass::new(MackeyGlassParams {
        a: 0.2,
        b: 0.1,
        n: 10,
        tau: 17,
        x0: 1.2,
        h: 1.0,
        steps: 4000,
        seed: Some(42),
        history: None,
    });
    let data_f64 = mg.generate();
    let data: Vec<f32> = data_f64.into_iter().map(|v| v as f32).collect();

    let inputs: Vec<Vec<f32>> = data[..data.len() - 1].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = data[1..].iter().map(|&v| vec![v]).collect();

    let mut esn = ESNBuilder::new(1) // 入力1次元
        .units(200)
        .spectral_radius(0.9)
        .seed(1)
        .build();
    esn.fit(&inputs, &targets, 1e-6);

    let preds: Vec<f32> = inputs
        .iter()
        .map(|u| esn.predict(&ndarray::array![u[0]]).to_vec()[0])
        .collect();

    let y_true: Vec<f32> = targets.iter().map(|v| v[0]).collect();
    println!("RMSE : {:.6}", rmse(&y_true, &preds));
    println!("R^2  : {:.6}", rsquare(&y_true, &preds));
}
