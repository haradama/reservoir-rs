#[cfg(feature = "alloc")]
extern crate alloc;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(not(feature = "alloc"))]
use core::marker::PhantomData;
#[cfg(feature = "alloc")]
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

impl_scalar_float!(f32, f64);

#[cfg(feature = "alloc")]
pub type State<S> = DVector<S>;
#[cfg(feature = "alloc")]
pub type Input<S> = DVector<S>;
#[cfg(feature = "alloc")]
pub type Output<S> = DVector<S>;
#[cfg(feature = "alloc")]
pub type Matrix<S> = DMatrix<S>;

#[cfg(not(feature = "alloc"))]
pub type State<S> = PhantomData<S>;
#[cfg(not(feature = "alloc"))]
pub type Input<S> = PhantomData<S>;
#[cfg(not(feature = "alloc"))]
pub type Output<S> = PhantomData<S>;
#[cfg(not(feature = "alloc"))]
pub type Matrix<S> = PhantomData<S>;

#[cfg(test)]
mod tests {
    use super::Scalar;

    const EPS_F64: f64 = 1e-12;
    const EPS_F32: f32 = 1e-6;

    #[test]
    fn test_scalar_f64_activation_abs_and_convert() {
        let x: f64 = 0.5;
        let y = x.activation();
        assert!((y - x.tanh()).abs() < EPS_F64);

        let z: f64 = (-3.0).abs_val();
        assert!((z - 3.0).abs() < EPS_F64);

        let a = f64::from_f64_val(1.25);
        assert!((a - 1.25).abs() < EPS_F64);
        assert!((a.to_f64_val() - 1.25).abs() < EPS_F64);
    }

    #[test]
    fn test_scalar_f32_activation_abs_and_convert() {
        let x: f32 = 0.5;
        let y = x.activation();
        assert!((y - x.tanh()).abs() < EPS_F32);

        let z: f32 = (-3.0).abs_val();
        assert!((z - 3.0).abs() < EPS_F32);

        let a = f32::from_f64_val(1.25);
        assert!((a - 1.25).abs() < EPS_F32);
        assert!((a.to_f64_val() - 1.25).abs() < 1e-6);
    }
}
