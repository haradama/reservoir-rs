use super::{DenseReservoir, RidgeReadout};
use reservoir_core::types::{Input, Output};
use reservoir_core::{Readout, Trainer, Reservoir};
use std::marker::PhantomData;

pub struct EchoStateNetwork {
    pub reservoir: DenseReservoir,
    pub readout: RidgeReadout,
    _marker: PhantomData<f32>,
}

impl EchoStateNetwork {
    pub fn predict(&mut self, input: &Input<f32>) -> Output<f32> {
        let state = self.reservoir.step(input);
        self.readout.predict(state)
    }
    pub fn fit(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], ridge: f32) {
        use crate::trainer::RidgeTrainer;
        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(&mut self.reservoir, &mut self.readout, inputs, targets)
            .unwrap();
    }
}

pub struct ESNBuilder {
    input_dim: usize,
    units: usize,
    spectral_radius: f32,
    seed: u64,
}
impl ESNBuilder {
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            units: 100,
            spectral_radius: 1.0,
            seed: 42,
        }
    }
    pub fn units(mut self, n: usize) -> Self {
        self.units = n;
        self
    }
    pub fn spectral_radius(mut self, r: f32) -> Self {
        self.spectral_radius = r;
        self
    }
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    pub fn build(self) -> EchoStateNetwork {
        EchoStateNetwork {
            reservoir: DenseReservoir::new(
                self.input_dim,
                self.units,
                self.spectral_radius,
                self.seed,
            ),
            readout: RidgeReadout::new(self.units),
            _marker: PhantomData,
        }
    }
}
