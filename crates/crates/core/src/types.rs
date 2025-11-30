extern crate alloc;
#[cfg(feature = "std")]
use nalgebra::{DMatrix, DVector};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use num_traits::{Float, One, Zero};

pub trait Scalar: Float + Zero + One + Copy + Send + Sync + core::fmt::Debug + 'static {}
impl<T> Scalar for T where T: Float + Zero + One + Copy + Send + Sync + core::fmt::Debug + 'static {}

#[cfg(feature = "std")]
pub type State<S>  = DVector<S>;
#[cfg(not(feature = "std"))]
pub type State<S>  = Vec<S>;

#[cfg(feature = "std")]
pub type Input<S>  = DVector<S>;
#[cfg(not(feature = "std"))]
pub type Input<S>  = Vec<S>;

#[cfg(feature = "std")]
pub type Output<S> = DVector<S>;
#[cfg(not(feature = "std"))]
pub type Output<S> = Vec<S>;

#[cfg(feature = "std")]
pub type Matrix<S> = DMatrix<S>;
#[cfg(not(feature = "std"))]
pub type Matrix<S> = ();
