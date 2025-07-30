use crate::float::RealScalar;
use crate::RidgeTrainer;

use super::{DenseReservoir, RidgeReadout};
use reservoir_core::types::{Input, Output};
use reservoir_core::{Readout, Reservoir, Trainer};

pub struct EchoStateNetwork<S: RealScalar> {
    pub reservoir: DenseReservoir<S>,
    pub readout: RidgeReadout<S>,
}

impl<S: RealScalar> EchoStateNetwork<S> {
    pub fn predict(&mut self, input: &Input<S>) -> Output<S> {
        let state = self.reservoir.step(input);
        self.readout.predict(state)
    }
    pub fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S) {
        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(&mut self.reservoir, &mut self.readout, inputs, targets)
            .unwrap();
    }
}

pub struct ESNBuilder<S: RealScalar> {
    input_dim: usize,
    units: usize,
    spectral_radius: S,
    input_scaling: S,
    leaking_rate: S,
    seed: u64,
    _m: std::marker::PhantomData<S>,
}

impl<S: RealScalar> ESNBuilder<S> {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            units: 100,
            spectral_radius: S::one(),
            input_scaling: S::one(),
            leaking_rate: S::one(),
            seed: 42,
            _m: std::marker::PhantomData,
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
    pub fn input_scaling(mut self, s: S) -> Self {
        self.input_scaling = s;
        self
    }
    pub fn leaking_rate(mut self, a: S) -> Self {
        self.leaking_rate = a;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    pub fn build(self) -> EchoStateNetwork<S> {
        let reservoir = DenseReservoir::new(
            self.input_dim,
            self.units,
            self.spectral_radius,
            self.input_scaling,
            self.leaking_rate,
            self.seed,
        );
        let readout = RidgeReadout::new(reservoir.dim());
        EchoStateNetwork { reservoir, readout }
    }
}
