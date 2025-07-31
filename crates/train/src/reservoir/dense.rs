use crate::float::RealScalar;
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{reservoir::Reservoir, types::*};

pub struct DenseReservoir<S: RealScalar> {
    w_in: DMatrix<S>,
    w: DMatrix<S>,
    leaking_rate: S,
    input_dim: usize,

    res_state: DVector<S>,
    ext_state: DVector<S>,
}

impl<S: RealScalar> DenseReservoir<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_dim: usize,
        units: usize,
        spectral_radius: S,
        input_scaling: S,
        leaking_rate: S,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let uni = Uniform::new(S::from(-0.5).unwrap(), S::from(0.5).unwrap());
        let mut rnd = |r: usize, c: usize| DMatrix::from_fn(r, c, |_, _| rng.sample(&uni));

        // 内部結合行列
        let mut w = rnd(units, units);
        let max_abs = w
            .iter()
            .fold(S::zero(), |m, &v| Float::max(m, Float::abs(v)));
        if max_abs != S::zero() {
            w /= max_abs;
            w *= spectral_radius;
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
        self.ext_state
            .rows_mut(1, self.input_dim)
            .copy_from(input);
        self.ext_state
            .rows_mut(1 + self.input_dim, self.res_state.len())
            .copy_from(&self.res_state);
    }
}

impl<S: RealScalar> Reservoir<S> for DenseReservoir<S> {
    fn reset(&mut self) {
        self.res_state.fill(S::zero());
        self.ext_state.fill(S::zero());
    }

    fn step(&mut self, input: &Input<S>) -> &State<S> {
        let pre = &self.w * &self.res_state + &self.w_in * input;
        let tanh = pre.map(|x| Float::tanh(x));
        self.res_state =
            &self.res_state * (S::one() - self.leaking_rate) + tanh * self.leaking_rate;

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
