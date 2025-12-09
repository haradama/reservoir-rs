use nalgebra::{DMatrix, DVector, Normed};
use rand::{distributions::Uniform, Rng, SeedableRng};
use reservoir_core::{reservoir::Reservoir, types::*};

use crate::RngType;

#[derive(Debug, Clone)]
pub struct DenseReservoir<S: Scalar> {
    w_in: DMatrix<S>,
    w: DMatrix<S>,
    leaking_rate: S,
    input_dim: usize,

    res_state: DVector<S>,
    ext_state: DVector<S>,
}

impl<S: Scalar> DenseReservoir<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_dim: usize,
        units: usize,
        spectral_radius: S,
        input_scaling: S,
        leaking_rate: S,
        seed: u64,
    ) -> Self {
        let mut rng = RngType::seed_from_u64(seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);
        let mut rnd =
            |r: usize, c: usize| DMatrix::from_fn(r, c, |_, _| S::from_f64_val(rng.sample(&uni)));

        let mut w = rnd(units, units);
        let w_f64 = w.map(|x| x.to_f64_val());
        let eigenvalues = w_f64.complex_eigenvalues();

        let current_rho = eigenvalues
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |a, b| a.max(b));

        let target_rho = spectral_radius.to_f64_val();

        if current_rho > 0.0 {
            let scale = target_rho / current_rho;
            w *= S::from_f64_val(scale);
        }

        let w_in = rnd(units, input_dim) * input_scaling;

        Self {
            w_in,
            w,
            leaking_rate,
            input_dim,
            res_state: DVector::zeros(units),
            ext_state: DVector::zeros(1 + input_dim + units),
        }
    }

    fn rebuild_ext_state(&mut self, input: &Input<S>) {
        self.ext_state[0] = S::one();
        self.ext_state.rows_mut(1, self.input_dim).copy_from(input);
        self.ext_state
            .rows_mut(1 + self.input_dim, self.res_state.len())
            .copy_from(&self.res_state);
    }
}

impl<S: Scalar> Reservoir<S> for DenseReservoir<S> {
    fn reset(&mut self) {
        self.res_state.fill(S::zero());
        self.ext_state.fill(S::zero());
    }

    fn step(&mut self, input: &Input<S>) -> &State<S> {
        let pre = &self.w * &self.res_state + &self.w_in * input;
        let act = pre.map(|x| x.activation());

        self.res_state = &self.res_state * (S::one() - self.leaking_rate) + act * self.leaking_rate;

        self.rebuild_ext_state(input);
        &self.ext_state
    }

    fn dim(&self) -> usize {
        self.ext_state.len()
    }

    fn state(&self) -> &State<S> {
        &self.ext_state
    }
}
