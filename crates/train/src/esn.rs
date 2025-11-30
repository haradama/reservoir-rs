use std::{marker::PhantomData, usize};

use crate::{
    float::RealScalar,
    input::IntoInput,
    readout::{LassoReadout, RidgeReadout},
    reservoir::DenseReservoir,
    trainer::{LassoTrainer, RidgeTrainer},
};
use nalgebra::DVector;
use reservoir_core::{types::Output, Input, Readout, Reservoir, Scalar, Trainer};

pub struct EchoStateNetwork<S: Scalar, R, O> {
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
    pub fn predict<I>(&mut self, input: I) -> Output<S>
    where
        I: IntoInput<S>,
    {
        let dv = input.into_dvector();
        let state = self.reservoir.step(&dv);
        self.readout.predict(state)
    }

    pub fn state_dim(&self) -> usize {
        self.reservoir.dim()
    }
}

impl<S> EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>>
where
    S: RealScalar,
{
    pub fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("training failed");
    }
}

impl<S> EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>>
where
    S: Scalar,
{
    pub fn fit_lasso(
        &mut self,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
        alpha: S,
        max_iter: usize,
        tol: S,
        washout: usize,
    ) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = LassoTrainer::new(alpha, max_iter, tol);
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("lasso training failed");
    }
}

pub struct ESNBuilder<S: Scalar> {
    input_dim: usize,
    output_dim: usize,
    units: usize,
    spectral_radius: S,
    input_scaling: S,
    leaking_rate: S,
    seed: u64,
    _marker: core::marker::PhantomData<S>,
}

impl<S: Scalar> ESNBuilder<S> {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            units: 100,
            spectral_radius: S::one(),
            input_scaling: S::one(),
            leaking_rate: S::one(),
            seed: 42,
            _marker: PhantomData,
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

    pub fn build(self) -> EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>> {
        let reservoir = DenseReservoir::new(
            self.input_dim,
            self.units,
            self.spectral_radius,
            self.input_scaling,
            self.leaking_rate,
            self.seed,
        );
        let readout = RidgeReadout::new(reservoir.dim(), self.output_dim, self.seed);

        EchoStateNetwork {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }

    pub fn build_lasso(self) -> EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>> {
        let reservoir = DenseReservoir::new(
            self.input_dim,
            self.units,
            self.spectral_radius,
            self.input_scaling,
            self.leaking_rate,
            self.seed,
        );
        let readout = LassoReadout::new(reservoir.dim(), self.output_dim, self.seed);

        EchoStateNetwork {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }
}
