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
        let n_sample = inputs.len();
        let dim_x = reservoir.dim();
        let mut x_mat = DMatrix::<f32>::zeros(n_sample, dim_x);
        let mut y_vec = DVector::<f32>::zeros(n_sample);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(&DVector::from_vec(u.clone())).clone_owned();
            x_mat.row_mut(i).copy_from(&state.transpose());
            y_vec[i] = t[0];
        }

        let gram = &x_mat.transpose() * &x_mat
            + DMatrix::<f32>::identity(dim_x, dim_x) * self.ridge;
        let w = (y_vec.transpose() * x_mat)
            * gram.try_inverse().expect("matrix inverse failed");
        readout.set_weights(DVector::from_row_slice(w.as_slice()));
        Ok(())
    }
}
