use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use rand::{SeedableRng, rngs::StdRng};

use reservoir_core::{reservoir::Reservoir, types::*};

pub struct DenseReservoir<S: Scalar + SampleUniform = f32> {
    w_in: Array2<S>,
    w: Array2<S>,
    state: Array1<S>,
}

impl<S: Scalar + SampleUniform> DenseReservoir<S> {
    pub fn new(input_dim: usize, units: usize, spectral_radius: S, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Uniform::new(S::from(-0.5).unwrap(), S::from(0.5).unwrap());

        let mut w = Array::random_using((units, units), &dist, &mut rng);
        let w_in = Array::random_using((units, input_dim), &dist, &mut rng);
        let max_abs = w.iter().fold(S::zero(), |m, v| m.max(v.abs()));
        if max_abs > S::zero() {
            w.mapv_inplace(|x| x * spectral_radius / max_abs);
        }

        Self {
            w_in,
            w,
            state: Array1::zeros(units),
        }
    }
}

impl<S: Scalar + SampleUniform> Reservoir<S> for DenseReservoir<S> {
    fn reset(&mut self) {
        self.state.fill(S::zero());
    }

    fn step(&mut self, input: &Input<S>) -> &State<S> {
        let pre = self.w.dot(&self.state) + self.w_in.dot(input);
        self.state = pre.mapv(|v| v.tanh());
        &self.state
    }

    fn dim(&self) -> usize {
        self.state.len()
    }

    fn state(&self) -> &State<S> {
        &self.state
    }
}
