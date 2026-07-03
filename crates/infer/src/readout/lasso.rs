//! Backwards-compatible alias for the linear readout.
//!
//! Historically this module exposed a `LassoReadout` type. LASSO regression is a
//! *training-time* choice, not a property of the readout (which is just a linear
//! map `y = W_out * state`), so the concrete type now lives in
//! [`crate::readout::linear`] as [`LinearReadout`].
//!
//! `LassoReadout` is kept as a type alias so existing code keeps compiling.

use crate::readout::linear::LinearReadout;

/// Linear readout produced by / used with LASSO training.
///
/// This is an alias for [`LinearReadout`]; prefer `LinearReadout` in new code.
pub type LassoReadout<S> = LinearReadout<S>;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use reservoir_core::readout::Readout;

    #[test]
    fn test_lasso_alias_predict() {
        let w_out = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 0.5]);
        let readout = LassoReadout::<f64>::create(w_out);
        let state = DVector::from_vec(vec![10.0, 2.0, 1.0]);
        assert_eq!(readout.predict(&state), DVector::from_vec(vec![17.0, 2.5]));
        assert_eq!(readout.output_dim(), 2);
    }
}
