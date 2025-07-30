use nalgebra::DVector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use reservoir_core::{readout::Readout, types::*};

pub struct RidgeReadout {
    w_out: DVector<f32>, // (units)
}

impl RidgeReadout {
    pub fn new(units: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(1234);
        let w_out = DVector::from_fn(units, |_, _| rng.gen::<f32>() - 0.5);
        Self { w_out }
    }
    pub fn set_weights(&mut self, w: DVector<f32>) {
        self.w_out = w;
    }
}

impl Readout<f32> for RidgeReadout {
    fn predict(&self, state: &State<f32>) -> Output<f32> {
        let y = self.w_out.dot(state);
        DVector::from_element(1, y)
    }
    fn output_dim(&self) -> usize {
        1
    }
}
