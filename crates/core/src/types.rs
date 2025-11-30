extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::{
    any::Any,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use cordic::CordicNumber;
use fixed::{types::extra::LeEqU32, FixedI32};

#[cfg(feature = "std")]
use nalgebra::{DMatrix, DVector};
use num_traits::{One, Zero};

pub trait Scalar:
    Copy
    + Send
    + Sync
    + core::fmt::Debug
    + 'static
    + Any
    + Zero
    + One
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
{
    fn activation(self) -> Self;
    fn from_f64_val(v: f64) -> Self;
    fn abs_val(self) -> Self;
}

macro_rules! impl_scalar_float {
    ($($t:ty),*) => {
        $(
            impl Scalar for $t {
                fn activation(self) -> Self {
                    self.tanh()
                }

                fn from_f64_val(v: f64) -> Self {
                    v as $t
                }

                fn abs_val(self) -> Self {
                    self.abs()
                }
            }
        )*
    };
}

impl_scalar_float!(f32, f64);

impl<Frac> Scalar for FixedI32<Frac>
where
    Frac: LeEqU32 + 'static + Send + Sync,
    FixedI32<Frac>: CordicNumber,
    FixedI32<Frac>: One,
{
    fn activation(self) -> Self {
        let one = <Self as One>::one();

        let threshold = Self::from_num(5);

        if self > threshold {
            return one;
        }
        if self < -threshold {
            return -one;
        }

        let two = one + one;
        let two_x = self * two;
        let e2x = cordic::exp(two_x);

        (e2x - one) / (e2x + one)
    }

    fn from_f64_val(v: f64) -> Self {
        Self::from_num(v)
    }

    fn abs_val(self) -> Self {
        self.abs()
    }
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
