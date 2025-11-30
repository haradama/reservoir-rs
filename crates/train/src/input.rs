use nalgebra::DVector;

use crate::float::RealScalar;

pub trait IntoInput<S: RealScalar> {
    fn into_dvector(self) -> DVector<S>;
}

impl<S: RealScalar> IntoInput<S> for Vec<S> {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_vec(self)
    }
}

impl<'a, S: RealScalar> IntoInput<S> for &'a [S] {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_row_slice(self)
    }
}

impl<S: RealScalar> IntoInput<S> for DVector<S> {
    fn into_dvector(self) -> DVector<S> {
        self
    }
}
