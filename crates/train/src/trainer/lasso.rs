use core::default::Default;
use core::result::Result;
use nalgebra::{DMatrix, DVector};
use reservoir_core::{
    reservoir::Reservoir,
    trainer::Trainer,
    types::{Input, Output},
    Scalar,
};
use reservoir_infer::LassoReadout;

pub struct LassoTrainer<S: Scalar> {
    pub alpha: S,
    pub max_iter: usize,
    pub tol: S,
}

impl<S: Scalar> Default for LassoTrainer<S> {
    fn default() -> Self {
        Self {
            alpha: S::from_f64_val(1e-4),
            max_iter: 1000,
            tol: S::from_f64_val(1e-4),
        }
    }
}

impl<S: Scalar> LassoTrainer<S> {
    pub fn new(alpha: S, max_iter: usize, tol: S) -> Self {
        Self {
            alpha,
            max_iter,
            tol,
        }
    }

    fn soft_threshold(z: S, alpha: S) -> S {
        if z > alpha {
            z - alpha
        } else if z < -alpha {
            z + alpha
        } else {
            S::zero()
        }
    }
}

impl<S, R> Trainer<R, LassoReadout<S>, S> for LassoTrainer<S>
where
    S: Scalar,
    R: Reservoir<S>,
{
    type Error = &'static str;

    fn fit(
        &mut self,
        reservoir: &mut R,
        readout: &mut LassoReadout<S>,
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

        let mut w_solved = DMatrix::<S>::zeros(dim_x, dim_y);

        for k in 0..dim_y {
            let mut w = DVector::<S>::zeros(dim_x);
            let y_corr = xty.column(k);

            for _ in 0..self.max_iter {
                let mut max_change = S::zero();

                for j in 0..dim_x {
                    let mut dot = S::zero();
                    for c in 0..dim_x {
                        dot += xtx[(j, c)] * w[c];
                    }

                    let rho = y_corr[j] - dot + xtx[(j, j)] * w[j];
                    let z_j = xtx[(j, j)];
                    let new_w_j = if z_j == S::zero() {
                        S::zero()
                    } else {
                        Self::soft_threshold(rho, self.alpha) / z_j
                    };

                    let change = (new_w_j - w[j]).abs_val();
                    if change > max_change {
                        max_change = change;
                    }
                    w[j] = new_w_j;
                }

                if max_change < self.tol {
                    break;
                }
            }
            w_solved.set_column(k, &w);
        }

        readout.set_weights(w_solved.transpose());
        Ok(())
    }
}
