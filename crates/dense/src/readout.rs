use nalgebra::DVector;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{readout::Readout, types::*};

use crate::float::RealScalar;

pub struct RidgeReadout<S: RealScalar> {
    w_out: DVector<S>,
}

impl<S: RealScalar> RidgeReadout<S> {
    pub fn new(dim: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        let uni = Uniform::new(S::from(-0.5).unwrap(), S::from(0.5).unwrap());
        let w_out = DVector::from_fn(dim, |_, _| rng.sample(&uni));
        Self { w_out }
    }
    pub fn set_weights(&mut self, w: DVector<S>) {
        self.w_out = w;
    }
}

impl<S: RealScalar> Readout<S> for RidgeReadout<S> {
    fn predict(&self, state: &State<S>) -> Output<S> {
        DVector::from_element(1, self.w_out.dot(state))
    }
    fn output_dim(&self) -> usize {
        1
    }
}
