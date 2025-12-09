use crate::{
    float::RealScalar,
    trainer::{LassoTrainer, RidgeTrainer},
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use nalgebra::{DMatrix, DVector, Normed};
use rand::{distributions::Uniform, Rng, SeedableRng};
use reservoir_core::{types::*, Input, Output, Reservoir, Trainer};
use reservoir_infer::{DenseReservoir, EchoStateNetwork, LassoReadout, RidgeReadout};

use crate::RngType;

pub trait ESNFitRidge<S>
where
    S: RealScalar,
{
    fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize);
}

impl<S> ESNFitRidge<S> for EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>>
where
    S: RealScalar,
{
    fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize) {
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

pub trait ESNFitLasso<S>
where
    S: Scalar,
{
    fn fit_lasso(
        &mut self,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
        alpha: S,
        max_iter: usize,
        tol: S,
        washout: usize,
    );
}

impl<S> ESNFitLasso<S> for EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>>
where
    S: Scalar,
{
    fn fit_lasso(
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
    _marker: PhantomData<S>,
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

    fn generate_reservoir_matrices(&self) -> (DMatrix<S>, DMatrix<S>) {
        let mut rng = RngType::seed_from_u64(self.seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);
        let mut rnd =
            |r: usize, c: usize| DMatrix::from_fn(r, c, |_, _| S::from_f64_val(rng.sample(&uni)));

        let mut w = rnd(self.units, self.units);

        let w_f64 = w.map(|x| x.to_f64_val());
        let eigenvalues = w_f64.complex_eigenvalues();

        let current_rho = eigenvalues
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |a, b| a.max(b));

        let target_rho = self.spectral_radius.to_f64_val();

        if current_rho > 0.0 {
            let scale = target_rho / current_rho;
            w *= S::from_f64_val(scale);
        }

        let w_in = rnd(self.units, self.input_dim) * self.input_scaling;

        (w_in, w)
    }

    fn generate_readout_matrix(&self, input_dim: usize, output_dim: usize) -> DMatrix<S> {
        let mut rng = RngType::seed_from_u64(self.seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);
        DMatrix::from_fn(output_dim, input_dim, |_, _| {
            S::from_f64_val(rng.sample(&uni))
        })
    }

    pub fn build(self) -> EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>> {
        let (w_in, w) = self.generate_reservoir_matrices();

        let reservoir =
            DenseReservoir::create(w_in, w, self.leaking_rate, self.input_dim, self.units);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);

        let readout = RidgeReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }

    pub fn build_lasso(self) -> EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>> {
        let (w_in, w) = self.generate_reservoir_matrices();

        let reservoir =
            DenseReservoir::create(w_in, w, self.leaking_rate, self.input_dim, self.units);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);

        let readout = LassoReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }
}
