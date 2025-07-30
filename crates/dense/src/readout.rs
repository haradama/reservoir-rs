use ndarray::{Array1, Array2};

use reservoir_core::{readout::Readout, types::*};

pub struct RidgeReadout<S: Scalar = f32> {
    w_out: Array2<S>,
}

impl<S: Scalar> RidgeReadout<S> {
    pub fn new(units: usize) -> Self {
        Self {
            w_out: Array2::zeros((1, units)),
        }
    }

    pub fn set_weights(&mut self, w: Array2<S>) {
        self.w_out = w;
    }
}

impl<S: Scalar> Readout<S> for RidgeReadout<S> {
    fn predict(&self, state: &State<S>) -> Output<S> {
        let y = self.w_out.dot(state);
        Array1::from_vec(y.to_vec())
    }

    fn output_dim(&self) -> usize {
        self.w_out.nrows()
    }
}
