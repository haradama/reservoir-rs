use nalgebra::{DMatrix, DVector};
use rand::{distributions::Standard, rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{reservoir::Reservoir, types::*};

pub struct DenseReservoir {
    w_in: DMatrix<f32>,
    w: DMatrix<f32>,
    leaking_rate: f32,
    input_dim: usize,

    res_state: DVector<f32>,
    ext_state: DVector<f32>,
}

impl DenseReservoir {
    pub fn new(
        input_dim: usize,
        units: usize,
        spectral_radius: f32,
        input_scaling: f32,
        leaking_rate: f32,
        seed: u64,
    ) -> Self {
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

        let w_in = rand_mat(units, input_dim) * input_scaling;

        Self {
            w_in,
            w,
            leaking_rate,
            input_dim,
            res_state: DVector::zeros(units),
            ext_state: DVector::zeros(1 + input_dim + units),
        }
    }

    fn build_ext_state(&mut self, input: &Input<f32>) {
        self.ext_state[0] = 1.0;
        self.ext_state.rows_mut(1, self.input_dim).copy_from(input);
        self.ext_state
            .rows_mut(1 + self.input_dim, self.res_state.len())
            .copy_from(&self.res_state);
    }
}

impl Reservoir<f32> for DenseReservoir {
    fn reset(&mut self) {
        self.res_state.fill(0.0);
        self.ext_state.fill(0.0);
    }

    fn step(&mut self, input: &Input<f32>) -> &State<f32> {
        let pre = &self.w * &self.res_state + &self.w_in * input;
        let tanh = pre.map(|x| x.tanh());
        self.res_state = (1.0 - self.leaking_rate) * &self.res_state + self.leaking_rate * tanh;

        self.build_ext_state(input);
        &self.ext_state
    }

    fn dim(&self) -> usize {
        self.ext_state.len()
    }

    fn state(&self) -> &State<f32> {
        &self.ext_state
    }
}
