use super::{DenseReservoir, RidgeReadout};
use crate::float::RealScalar;
use nalgebra::{DMatrix, DVector};
use reservoir_core::{reservoir::Reservoir, trainer::Trainer};

pub struct RidgeTrainer<S: RealScalar> {
    pub ridge: S,
}

impl<S: RealScalar> Default for RidgeTrainer<S> {
    fn default() -> Self {
        Self {
            ridge: S::from(1e-6).unwrap(),
        }
    }
}

impl<S: RealScalar> Trainer<DenseReservoir<S>, RidgeReadout<S>, S> for RidgeTrainer<S> {
    type Error = ();

    fn fit(
        &mut self,
        reservoir: &mut DenseReservoir<S>,
        readout: &mut RidgeReadout<S>,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
    ) -> Result<(), Self::Error> {
        let n = inputs.len();
        let dim_x = reservoir.dim();
        let mut x_mat = DMatrix::<S>::zeros(n, dim_x);
        let mut y_vec = DVector::<S>::zeros(n);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(&DVector::from_vec(u.clone())).clone_owned();
            x_mat.row_mut(i).copy_from(&state.transpose());
            y_vec[i] = t[0];
        }

        let gram = &x_mat.transpose() * &x_mat + DMatrix::<S>::identity(dim_x, dim_x) * self.ridge;
        let w = (y_vec.transpose() * x_mat) * gram.try_inverse().expect("Gram inverse failed");
        readout.set_weights(DVector::from_row_slice(w.as_slice()));
        Ok(())
    }
}
