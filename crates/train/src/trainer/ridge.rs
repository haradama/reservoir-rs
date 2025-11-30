use crate::{float::RealScalar, readout::RidgeReadout, reservoir::DenseReservoir};
use nalgebra::{DMatrix};
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
        if inputs.is_empty() {
            return Err("empty inputs");
        }

        let n_samples = inputs.len();
        let dim_x = reservoir.dim();
        let dim_y = targets[0].len();

        let mut x_mat = DMatrix::<S>::zeros(n_samples, dim_x);
        let mut y_mat = DMatrix::<S>::zeros(n_samples, dim_y);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(u);
            for j in 0..dim_x {
                x_mat[(i, j)] = state[j];
            }
            for j in 0..dim_y {
                y_mat[(i, j)] = t[j];
            }
        }

        let gram = &x_mat.transpose() * &x_mat + DMatrix::<S>::identity(dim_x, dim_x) * self.ridge;
        let rhs = &x_mat.transpose() * &y_mat;

        let gram_chol = gram.cholesky().ok_or("Cholesky failed")?;
        let w_solved = gram_chol.solve(&rhs);

        readout.set_weights(w_solved.transpose());
        Ok(())
    }
}
