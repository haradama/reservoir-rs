use crate::RidgeTrainer;

use super::{DenseReservoir, RidgeReadout};
use reservoir_core::types::{Input, Output};
use reservoir_core::{Readout, Trainer, Reservoir};
use std::marker::PhantomData;

pub struct EchoStateNetwork {
    pub reservoir: DenseReservoir,
    pub readout:   RidgeReadout,
    _marker:  PhantomData<f32>,
}

impl EchoStateNetwork {
    pub fn predict(&mut self, input: &Input<f32>) -> Output<f32> {
        let state = self.reservoir.step(input);
        self.readout.predict(state)
    }
    pub fn fit(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], ridge: f32) {
        let mut trainer = RidgeTrainer { ridge };
        trainer.fit(&mut self.reservoir, &mut self.readout, inputs, targets).unwrap();
    }
}

pub struct ESNBuilder {
    input_dim:      usize,
    units:          usize,
    spectral_radius:f32,
    input_scaling:  f32,
    leaking_rate:   f32,
    seed:           u64,
}
impl ESNBuilder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            units: 100,
            spectral_radius: 1.0,
            input_scaling: 1.0,
            leaking_rate: 1.0,
            seed: 42,
        }
    }
    pub fn units(mut self, n: usize)            -> Self { self.units = n; self }
    pub fn spectral_radius(mut self, r: f32)    -> Self { self.spectral_radius = r; self }
    pub fn input_scaling  (mut self, s: f32)    -> Self { self.input_scaling   = s; self }
    pub fn leaking_rate   (mut self, a: f32)    -> Self { self.leaking_rate    = a; self }
    pub fn seed           (mut self, s: u64)    -> Self { self.seed            = s; self }

    pub fn build(self) -> EchoStateNetwork {
        let reservoir = DenseReservoir::new(
            self.input_dim,
            self.units,
            self.spectral_radius,
            self.input_scaling,
            self.leaking_rate,
            self.seed,
        );
        let readout   = RidgeReadout::new(reservoir.dim());
        EchoStateNetwork { reservoir, readout, _marker: PhantomData }
    }
}
