use nalgebra::DMatrix;
use reservoir_core::{readout::Readout, types::*};

#[derive(Debug, Clone)]
pub struct RidgeReadout<S: Scalar> {
    pub w_out: DMatrix<S>,
    pub output_dim: usize,
}

impl<S: Scalar> RidgeReadout<S> {
    pub fn create(w_out: DMatrix<S>) -> Self {
        let output_dim = w_out.nrows();
        Self { w_out, output_dim }
    }

    pub fn set_weights(&mut self, w: DMatrix<S>) {
        self.output_dim = w.nrows();
        self.w_out = w;
    }

    pub fn weights(&self) -> &DMatrix<S> {
        &self.w_out
    }
}

impl<S: Scalar> Readout<S> for RidgeReadout<S> {
    fn predict(&self, state: &State<S>) -> Output<S> {
        &self.w_out * state
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}
