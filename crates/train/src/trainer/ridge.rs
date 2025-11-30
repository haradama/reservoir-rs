use crate::{float::RealScalar, readout::RidgeReadout, reservoir::DenseReservoir};
use nalgebra::DMatrix;
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
            ridge: S::from_f64_val(1e-6),
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
        washout: usize,
    ) -> Result<(), Self::Error> {
        if inputs.len() != targets.len() {
            return Err("inputs and targets length mismatch");
        }
        if inputs.is_empty() {
            return Err("empty inputs");
        }
        if washout >= inputs.len() {
            return Err("washout period is larger than or equal to input length");
        }

        let dim_x = reservoir.dim();
        let dim_y = targets[0].len();

        let mut xtx = DMatrix::<S>::zeros(dim_x, dim_x);
        let mut xty = DMatrix::<S>::zeros(dim_x, dim_y);

        for (i, (u, t)) in inputs.iter().zip(targets).enumerate() {
            let state = reservoir.step(u);

            if i >= washout {
                xtx.ger(S::one(), state, state, S::one());
                xty.ger(S::one(), state, t, S::one());
            }
        }

        for i in 0..dim_x {
            xtx[(i, i)] = xtx[(i, i)] + self.ridge;
        }

        let gram_chol = xtx
            .cholesky()
            .ok_or("Cholesky failed (matrix might not be positive definite)")?;
        let w_solved = gram_chol.solve(&xty);

        readout.set_weights(w_solved.transpose());
        Ok(())
    }
}
