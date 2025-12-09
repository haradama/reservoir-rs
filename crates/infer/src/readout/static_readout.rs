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
