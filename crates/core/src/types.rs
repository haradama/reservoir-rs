extern crate alloc;
use core::{
    any::Any,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[cfg(any(feature = "std", feature = "libm"))]
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
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
{
    fn activation(self) -> Self;
    fn from_f64_val(v: f64) -> Self;
    fn to_f64_val(self) -> f64;
    fn abs_val(self) -> Self;
}

#[cfg(any(feature = "std", feature = "libm"))]
macro_rules! impl_scalar_float {
    ($($t:ty),*) => {
        $(
            impl Scalar for $t {
                fn activation(self) -> Self {
                    num_traits::Float::tanh(self)
                }

                fn from_f64_val(v: f64) -> Self {
                    v as $t
                }

                fn to_f64_val(self) -> f64 {
                    self as f64
                }

                fn abs_val(self) -> Self {
                    num_traits::Float::abs(self)
                }
            }
        )*
    };
}

#[cfg(any(feature = "std", feature = "libm"))]
impl_scalar_float!(f32, f64);

#[cfg(any(feature = "std", feature = "libm"))]
pub type State<S> = DVector<S>;
#[cfg(any(feature = "std", feature = "libm"))]
pub type Input<S> = DVector<S>;
#[cfg(any(feature = "std", feature = "libm"))]
pub type Output<S> = DVector<S>;
#[cfg(any(feature = "std", feature = "libm"))]
pub type Matrix<S> = DMatrix<S>;

#[cfg(not(any(feature = "std", feature = "libm")))]
pub type State<S> = PhantomData<S>;
#[cfg(not(any(feature = "std", feature = "libm")))]
pub type Input<S> = PhantomData<S>;
#[cfg(not(any(feature = "std", feature = "libm")))]
pub type Output<S> = PhantomData<S>;
#[cfg(not(any(feature = "std", feature = "libm")))]
pub type Matrix<S> = PhantomData<S>;
