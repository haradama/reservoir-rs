use crate::esn::StaticReadout as StaticReadoutTrait;
use nalgebra::{SMatrix, SVector};
use reservoir_core::types::Scalar;

#[derive(Debug, Clone)]
pub struct StaticReadout<S: Scalar, const IN_DIM: usize, const OUT_DIM: usize> {
    pub w_out: SMatrix<S, OUT_DIM, IN_DIM>,
}

impl<S: Scalar, const IN_DIM: usize, const OUT_DIM: usize> StaticReadout<S, IN_DIM, OUT_DIM> {
    pub fn create(w_out: SMatrix<S, OUT_DIM, IN_DIM>) -> Self {
        Self { w_out }
    }

    pub fn predict(&self, state: &SVector<S, IN_DIM>) -> SVector<S, OUT_DIM> {
        self.w_out * state
    }
}

impl<S: Scalar, const IN: usize, const OUT: usize> StaticReadoutTrait<S, IN, OUT>
    for StaticReadout<S, IN, OUT>
{
    fn predict_static(&self, state: &SVector<S, IN>) -> SVector<S, OUT> {
        self.predict(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SMatrix, SVector};

    #[test]
    fn test_static_readout_predict() {
        let w: SMatrix<f32, 2, 3> = SMatrix::from_row_slice(&[1.0, 2.0, 3.0, 0.0, 1.0, 0.5]);
        let r = StaticReadout::<f32, 3, 2>::create(w);

        let x = SVector::<f32, 3>::new(10.0, 2.0, 1.0);
        let y = r.predict(&x);

        assert_eq!(y, SVector::<f32, 2>::new(17.0, 2.5));
    }
}
