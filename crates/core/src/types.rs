use nalgebra::{DMatrix, DVector};
use num_traits::{Float, One, Zero};

pub trait Scalar:
    Float + Zero + One + Copy + Send + Sync + std::fmt::Debug + 'static
{
}
impl<T> Scalar for T where
    T: Float + Zero + One + Copy + Send + Sync + std::fmt::Debug + 'static
{
}

pub type State<S>  = DVector<S>;
pub type Input<S>  = DVector<S>;
pub type Output<S> = DVector<S>;
pub type Matrix<S> = DMatrix<S>;
