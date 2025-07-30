use nalgebra::{DMatrix, DVector};
use rand::{distributions::Standard, rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{reservoir::Reservoir, types::*};

pub struct DenseReservoir {
    w_in: DMatrix<f32>,
    w: DMatrix<f32>,
    state: DVector<f32>,
}

impl DenseReservoir {
    pub fn new(input_dim: usize, units: usize, spectral_radius: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rand_mat = |r: usize, c: usize| {
            DMatrix::from_fn(r, c, |_, _| rng.sample::<f32, _>(Standard) - 0.5)
        };

        let mut w = rand_mat(units, units);
        let max_abs = w.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs > 0.0 {
            w /= max_abs;
            w *= spectral_radius;
        }
        let w_in = rand_mat(units, input_dim);

        Self {
            w_in,
            w,
            state: DVector::zeros(units),
        }
    }
}

impl Reservoir<f32> for DenseReservoir {
    fn reset(&mut self) {
        self.state.fill(0.0);
    }

    fn step(&mut self, input: &Input<f32>) -> &State<f32> {
        self.state = &self.w * &self.state + &self.w_in * input;
        self.state.apply(|x| *x = x.tanh());
        &self.state
    }
    fn dim(&self) -> usize {
        self.state.len()
    }
    fn state(&self) -> &State<f32> {
        &self.state
    }
}
