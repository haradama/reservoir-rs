use crate::float::RealScalar;
use core::default::Default;
use core::result::Result;
use nalgebra::DMatrix;
use reservoir_core::{
    reservoir::Reservoir,
    trainer::Trainer,
    types::{Input, Output},
};
use reservoir_infer::RidgeReadout;

/// Ridge regression trainer for linear readouts.
///
/// This trainer accumulates:
/// - `X^T X` and `X^T Y` from reservoir states (after `washout`)
/// then solves:
/// - `(X^T X + ridge * I) W = X^T Y`
///
/// The readout stores weights as `W_out` (output_dim x state_dim),
/// so the solved matrix is transposed before being set.
///
/// # Errors
/// Returns static string errors for common input validation failures or
/// linear solve failures.
pub struct RidgeTrainer<S: RealScalar> {
    /// L2 regularization added to diagonal of `X^T X`.
    pub ridge: S,
}

impl<S: RealScalar> Default for RidgeTrainer<S> {
    fn default() -> Self {
        Self {
            ridge: S::from_f64_val(1e-6),
        }
    }
}

impl<S, R> Trainer<R, RidgeReadout<S>, S> for RidgeTrainer<S>
where
    S: RealScalar,
    R: Reservoir<S>,
{
    type Error = &'static str;

    fn fit(
        &mut self,
        reservoir: &mut R,
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
            xtx[(i, i)] += self.ridge;
        }

        let w_solved = xtx
            .lu()
            .solve(&xty)
            .ok_or("Linear resolution failed (LU decomposition)")?;

        readout.set_weights(w_solved.transpose());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use reservoir_core::types::{Input, State};
    use reservoir_core::Reservoir;
    use reservoir_infer::RidgeReadout;

    #[derive(Debug, Clone)]
    struct LinearReservoir {
        input_dim: usize,
        ext: DVector<f64>, // [1, u...]
    }

    impl LinearReservoir {
        fn new(input_dim: usize) -> Self {
            Self {
                input_dim,
                ext: DVector::zeros(1 + input_dim),
            }
        }
    }

    impl Reservoir<f64> for LinearReservoir {
        fn reset(&mut self) {
            self.ext.fill(0.0);
        }
        fn step(&mut self, input: &Input<f64>) -> &State<f64> {
            self.ext[0] = 1.0;
            self.ext.rows_mut(1, self.input_dim).copy_from(input);
            &self.ext
        }
        fn dim(&self) -> usize {
            self.ext.len()
        }
        fn state(&self) -> &State<f64> {
            &self.ext
        }
    }

    #[test]
    fn test_ridge_trainer_fit_recovers_linear_weights() {
        // y = 2 + 3u, state = [1, u]
        let mut reservoir = LinearReservoir::new(1);

        let mut readout = RidgeReadout::<f64>::create(DMatrix::zeros(1, reservoir.dim()));

        let us = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let inputs: Vec<DVector<f64>> = us.iter().map(|&u| DVector::from_vec(vec![u])).collect();
        let targets: Vec<DVector<f64>> = us
            .iter()
            .map(|&u| DVector::from_vec(vec![2.0 + 3.0 * u]))
            .collect();

        let mut trainer = RidgeTrainer { ridge: 1e-12 };
        trainer
            .fit(&mut reservoir, &mut readout, &inputs, &targets, 0)
            .unwrap();

        // readout.w_out: (1 x 2) = [bias, slope]
        let w = readout.weights();
        assert_eq!(w.nrows(), 1);
        assert_eq!(w.ncols(), 2);

        let bias = w[(0, 0)];
        let slope = w[(0, 1)];

        assert!((bias - 2.0).abs() < 1e-6);
        assert!((slope - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_ridge_trainer_errors() {
        let mut reservoir = LinearReservoir::new(1);
        let mut readout = RidgeReadout::<f64>::create(DMatrix::zeros(1, reservoir.dim()));
        let mut trainer = RidgeTrainer { ridge: 1e-6 };

        let inputs: Vec<DVector<f64>> = vec![];
        let targets: Vec<DVector<f64>> = vec![];
        assert_eq!(
            trainer
                .fit(&mut reservoir, &mut readout, &inputs, &targets, 0)
                .unwrap_err(),
            "empty inputs"
        );

        let inputs = vec![DVector::from_vec(vec![0.0])];
        let targets = vec![];
        assert_eq!(
            trainer
                .fit(&mut reservoir, &mut readout, &inputs, &targets, 0)
                .unwrap_err(),
            "inputs and targets length mismatch"
        );

        let inputs = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];
        let targets = vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![1.0])];
        assert_eq!(
            trainer
                .fit(&mut reservoir, &mut readout, &inputs, &targets, 2)
                .unwrap_err(),
            "washout period is larger than or equal to input length"
        );
    }
}
