use crate::{float::RealScalar, readout::RidgeReadout, reservoir::DenseReservoir};
use nalgebra::{DMatrix, DVector};
use reservoir_core::{
    reservoir::Reservoir,
    trainer::Trainer,
    types::{Input, Output},
};

pub struct RidgeTrainer<S: RealScalar> {
    pub ridge: S,
}

impl<S: RealScalar> Default for RidgeTrainer<S> {
    fn default() -> Self {
        Self {
            ridge: S::from_f64(1e-6).expect("from_f64 failed"),
        }
    }
}

impl<S: RealScalar> Trainer<DenseReservoir<S>, RidgeReadout<S>, S> for RidgeTrainer<S> {
    type Error = &'static str;

    fn fit(
        &mut self,
        reservoir: &mut DenseReservoir<S>,
        readout: &mut RidgeReadout<S>,
        inputs: &[Input<S>],
        targets: &[Output<S>],
    ) -> Result<(), Self::Error> {
        if inputs.len() != targets.len() {
            return Err("inputs and targets length mismatch");
        }

        let n = inputs.len();
        let dim_x = reservoir.dim();

        let mut x_mat = DMatrix::<S>::zeros(n, dim_x);
        let mut y_vec = DVector::<S>::zeros(n);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(u).clone_owned();
            x_mat.row_mut(i).copy_from(&state.transpose());
            y_vec[i] = t[0];
        }

        let gram = &x_mat.transpose() * &x_mat + DMatrix::<S>::identity(dim_x, dim_x) * self.ridge;
        let rhs  = y_vec.transpose() * x_mat;
        let gram_chol = gram.cholesky().ok_or("Cholesky failed")?;
        let w_row = gram_chol.solve(&rhs.transpose());

        readout.set_weights(w_row);
        Ok(())
    }
}
