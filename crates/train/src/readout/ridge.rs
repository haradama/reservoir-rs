use nalgebra::DMatrix;
use rand::{distributions::Uniform, rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{readout::Readout, types::*};

#[derive(Debug, Clone)]
pub struct RidgeReadout<S: Scalar> {
    w_out: DMatrix<S>,
    output_dim: usize,
}

impl<S: Scalar> RidgeReadout<S> {
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let uni = Uniform::new(-0.5f64, 0.5f64);
        let w_out = DMatrix::from_fn(output_dim, input_dim, |_, _| {
            S::from_f64_val(rng.sample(&uni))
        });

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
