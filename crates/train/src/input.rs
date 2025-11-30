use nalgebra::DVector;
use reservoir_core::types::Scalar;

pub trait IntoInput<S: Scalar> {
    fn into_dvector(self) -> DVector<S>;
}

impl<S: Scalar> IntoInput<S> for Vec<S> {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_vec(self)
    }
}

impl<'a, S: Scalar> IntoInput<S> for &'a [S] {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_row_slice(self)
    }
}

impl<S: Scalar> IntoInput<S> for DVector<S> {
    fn into_dvector(self) -> DVector<S> {
        self
    }
}
