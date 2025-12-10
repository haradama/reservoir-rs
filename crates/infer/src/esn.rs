use core::marker::PhantomData;
#[cfg(feature = "alloc")]
use reservoir_core::types::Output;
use reservoir_core::Scalar;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use heapless::Vec as HVec;
#[cfg(feature = "alloc")]
use nalgebra::DVector;
#[cfg(feature = "alloc")]
use reservoir_core::Readout;
#[cfg(feature = "alloc")]
use reservoir_core::Reservoir;

use nalgebra::SVector;

#[cfg(feature = "alloc")]
pub trait IntoInput<S: Scalar> {
    fn into_dvector(self) -> DVector<S>;
}

#[cfg(feature = "alloc")]
impl<S: Scalar> IntoInput<S> for Vec<S> {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_vec(self)
    }
}

#[cfg(feature = "alloc")]
impl<'a, S: Scalar> IntoInput<S> for &'a [S] {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_row_slice(self)
    }
}

#[cfg(feature = "alloc")]
impl<S: Scalar> IntoInput<S> for DVector<S> {
    fn into_dvector(self) -> DVector<S> {
        self
    }
}

#[cfg(feature = "alloc")]
impl<S: Scalar, const N: usize> IntoInput<S> for HVec<S, N> {
    fn into_dvector(self) -> DVector<S> {
        DVector::from_iterator(self.len(), self.into_iter())
    }
}

#[cfg(feature = "alloc")]
pub struct EchoStateNetwork<S: Scalar, R, O> {
    pub reservoir: R,
    pub readout: O,
    _marker: PhantomData<S>,
}

#[cfg(feature = "alloc")]
impl<S, R, O> EchoStateNetwork<S, R, O>
where
    S: Scalar,
    R: Reservoir<S>,
    O: Readout<S>,
{
    pub fn new(reservoir: R, readout: O) -> Self {
        Self {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }

    pub fn predict<I>(&mut self, input: I) -> Output<S>
    where
        I: IntoInput<S>,
    {
        let dv = input.into_dvector();
        let state = self.reservoir.step(&dv);
        self.readout.predict(state)
    }

    pub fn state_dim(&self) -> usize {
        self.reservoir.dim()
    }
}

pub trait StaticReservoir<S: Scalar, const IN: usize, const EXT: usize> {
    fn step_static(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT>;
}

pub trait StaticReadout<S: Scalar, const H: usize, const OUT: usize> {
    fn predict_static(&self, state: &SVector<S, H>) -> SVector<S, OUT>;
}

pub struct StaticESN<S, R, O> {
    pub reservoir: R,
    pub readout: O,
    _marker: PhantomData<S>,
}

impl<S, R, O> StaticESN<S, R, O>
where
    S: Scalar,
{
    pub fn new(reservoir: R, readout: O) -> Self {
        Self {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }

    pub fn predict<const IN: usize, const OUT: usize, const HIDDEN: usize>(
        &mut self,
        input: &SVector<S, IN>,
    ) -> SVector<S, OUT>
    where
        R: StaticReservoir<S, IN, HIDDEN>,
        O: StaticReadout<S, HIDDEN, OUT>,
    {
        let state = self.reservoir.step_static(input);
        self.readout.predict_static(state)
    }
}
