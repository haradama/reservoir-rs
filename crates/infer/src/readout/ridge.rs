use nalgebra::DMatrix;
use reservoir_core::{readout::Readout, types::*};

#[derive(Debug, Clone)]
pub struct RidgeReadout<S: Scalar> {
    pub w_out: DMatrix<S>,
    pub output_dim: usize,
}

impl<S: Scalar> RidgeReadout<S> {
    pub fn create(w_out: DMatrix<S>) -> Self {
        let output_dim = w_out.nrows();
        Self { w_out, output_dim }
    }

    pub fn set_weights(&mut self, w: DMatrix<S>) {
        self.output_dim = w.nrows();
        self.w_out = w;
    }

    pub fn weights(&self) -> &DMatrix<S> {
        &self.w_out
    }
}

impl<S: Scalar> Readout<S> for RidgeReadout<S> {
    fn predict(&self, state: &State<S>) -> Output<S> {
        &self.w_out * state
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    type TestScalar = f64;

    #[test]
    fn test_ridge_readout_creation() {
        let w_out = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let readout = RidgeReadout::<TestScalar>::create(w_out.clone());
        assert_eq!(readout.output_dim(), 2);
        assert_eq!(readout.weights(), &w_out);
    }

    #[test]
    fn test_ridge_readout_predict() {
        let w_out = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 0.0, 1.0, 0.5]);

        let readout = RidgeReadout::<TestScalar>::create(w_out);
        let state = DVector::from_vec(vec![10.0, 2.0, 1.0]);
        let expected_output = DVector::from_vec(vec![17.0, 2.5]);
        let prediction = readout.predict(&state);
        assert_eq!(prediction.len(), 2);
        assert_eq!(prediction, expected_output);
    }

    #[test]
    fn test_set_weights() {
        let initial_w = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
        let mut readout = RidgeReadout::<TestScalar>::create(initial_w);

        assert_eq!(readout.output_dim(), 1);

        let new_w = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);

        readout.set_weights(new_w.clone());
        assert_eq!(readout.output_dim(), 3);
        assert_eq!(readout.weights(), &new_w);
    }
}
