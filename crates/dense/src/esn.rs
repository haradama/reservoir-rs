use std::marker::PhantomData;

use super::{DenseReservoir, RidgeReadout};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use reservoir_core::{readout::Readout, reservoir::Reservoir, trainer::Trainer, types::*};

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
    pub fn predict(&mut self, input: &Input<S>) -> Output<S> {
        let state = self.reservoir.step(input);
        self.readout.predict(state)
    }
}

impl EchoStateNetwork<f32, DenseReservoir<f32>, RidgeReadout<f32>> {
    pub fn fit(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], ridge: f32) {
        use crate::trainer::RidgeTrainer;
        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(&mut self.reservoir, &mut self.readout, inputs, targets)
            .unwrap();
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
        EchoStateNetwork {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }
}
