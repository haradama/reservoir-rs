use nalgebra::{DMatrix, DVector};
use reservoir_core::{reservoir::Reservoir, trainer::Trainer};

use super::{DenseReservoir, RidgeReadout};

pub struct RidgeTrainer {
    pub ridge: f32,
}
impl Default for RidgeTrainer {
    fn default() -> Self {
        Self { ridge: 1e-6 }
    }
}

impl Trainer<DenseReservoir, RidgeReadout, f32> for RidgeTrainer {
    type Error = ();

    fn fit(
        &mut self,
        reservoir: &mut DenseReservoir,
        readout: &mut RidgeReadout,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<(), Self::Error> {
        let n = inputs.len();
        let units = reservoir.dim();

        let mut x_mat = DMatrix::<f32>::zeros(n, units);
        let mut y_vec = DVector::<f32>::zeros(n);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(&DVector::from_vec(u.clone())).clone_owned();
            x_mat.row_mut(i).copy_from(&state.transpose());
            y_vec[i] = t[0];
        }

        let gram =
            &x_mat.transpose() * &x_mat + DMatrix::<f32>::identity(units, units) * self.ridge;
        let inv = gram.try_inverse().unwrap();
        let w = (y_vec.transpose() * x_mat) * inv; // Row Vec
        readout.set_weights(DVector::from_row_slice(w.as_slice()));
        Ok(())
    }
}
