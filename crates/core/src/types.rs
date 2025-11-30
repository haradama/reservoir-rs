extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ops::{Add, Div, Mul, Sub};
#[cfg(feature = "std")]
use nalgebra::{DMatrix, DVector};
use num_traits::{One, Zero};

pub trait Scalar:
    Copy
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
}

impl<T> Scalar for T where
    T: Copy
        + Send
        + Sync
        + core::fmt::Debug
        + 'static
        + Zero
        + One
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
{
}

#[cfg(feature = "std")]
pub type State<S> = DVector<S>;
#[cfg(not(feature = "std"))]
pub type State<S> = Vec<S>;

#[cfg(feature = "std")]
pub type Input<S> = DVector<S>;
#[cfg(not(feature = "std"))]
pub type Input<S> = Vec<S>;

#[cfg(feature = "std")]
pub type Output<S> = DVector<S>;
#[cfg(not(feature = "std"))]
pub type Output<S> = Vec<S>;

#[cfg(feature = "std")]
pub type Matrix<S> = DMatrix<S>;
#[cfg(not(feature = "std"))]
pub type Matrix<S> = ();
