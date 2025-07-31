#![allow(clippy::missing_docs_in_private_items)]

extern crate alloc;

use alloc::vec::Vec;

pub trait Scalar: Copy + core::fmt::Debug + 'static + Send + Sync {}
impl<T> Scalar for T where T: Copy + core::fmt::Debug + 'static + Send + Sync {}

pub trait Reservoir {
    type Scalar: Scalar;

    type State<'a>: AsRef<[Self::Scalar]>
    where
        Self: 'a;

    fn reset(&mut self);
    fn step(&mut self, input: &[Self::Scalar]) -> Self::State<'_>;
    fn dim(&self) -> usize;
}

pub trait Readout {
    type Scalar: Scalar;
    fn predict(&self, state: &[Self::Scalar]) -> Self::Scalar;
}

#[cfg(feature = "alloc")]
mod trainer_trait {
    use super::{Readout, Reservoir};
    use alloc::vec::Vec;

    pub trait Trainer<R, O>
    where
        R: Reservoir<Scalar = O::Scalar>,
        O: Readout,
    {
        type Error;

        fn fit(
            &mut self,
            reservoir: &mut R,
            readout: &mut O,
            inputs: &[Vec<O::Scalar>],
            targets: &[Vec<O::Scalar>],
        ) -> Result<(), Self::Error>;
    }
}

#[cfg(feature = "alloc")]
pub use trainer_trait::Trainer;
