use ndarray::{Array1, Array2};
use num_traits::{Float, One, Zero};

pub trait Scalar: Float + Zero + One + Copy + Send + Sync + std::fmt::Debug + 'static {}
impl<T> Scalar for T where T: Float + Zero + One + Copy + Send + Sync + std::fmt::Debug + 'static {}

pub type State<S> = Array1<S>;
pub type Input<S> = Array1<S>;
pub type Output<S> = Array1<S>;

pub type Matrix<S> = Array2<S>;
