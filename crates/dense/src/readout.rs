use ndarray::{Array1, Array2};

use rand::{rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{readout::Readout, types::*};

pub struct RidgeReadout<S: Scalar = f32> {
    w_out: Array2<S>,
}

impl<S: Scalar> RidgeReadout<S> {
    pub fn new(units: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        let mut w = Array2::<S>::zeros((1, units));
        for v in w.iter_mut() {
            *v = S::from(rng.gen::<f32>() - 0.5).unwrap();
        }
        Self { w_out: w }
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
