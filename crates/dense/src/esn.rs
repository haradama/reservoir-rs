use std::marker::PhantomData;

use super::{DenseReservoir, RidgeReadout};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use reservoir_core::{readout::Readout, reservoir::Reservoir, types::*};

pub struct EchoStateNetwork<S = f32, R = DenseReservoir<S>, O = RidgeReadout<S>> {
    pub reservoir: R,
    pub readout: O,
    _marker: PhantomData<S>,
}

impl<S, R, O> EchoStateNetwork<S, R, O>
where
    S: Scalar,
    R: Reservoir<S>,
    O: Readout<S>,
{
    pub fn predict(&mut self, u: &Input<S>) -> Output<S> {
        let state = self.reservoir.step(u);
        self.readout.predict(state)
    }
}

pub struct ESNBuilder<S: Scalar = f32> {
    input_dim: usize,
    units: usize,
    spectral_radius: S,
    seed: u64,
}

impl<S: Scalar> ESNBuilder<S> {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            units: 100,
            spectral_radius: S::one(),
            seed: 42,
        }
    }

    pub fn units(mut self, n: usize) -> Self {
        self.units = n;
        self
    }

    pub fn spectral_radius(mut self, r: S) -> Self {
        self.spectral_radius = r;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

impl<S> ESNBuilder<S>
where
    S: Scalar + SampleUniform,
{
    pub fn build(self) -> EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>> {
        let reservoir =
            DenseReservoir::new(self.input_dim, self.units, self.spectral_radius, self.seed);
        let readout = RidgeReadout::new(self.units);
        EchoStateNetwork::<S, DenseReservoir<S>, RidgeReadout<S>> {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }
}
