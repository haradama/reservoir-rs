use alloc::vec::Vec;
use core::marker::PhantomData;
use nalgebra::DVector;
use reservoir_core::Readout;
use reservoir_core::Reservoir;
use reservoir_core::{types::Output, Scalar};

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

pub struct EchoStateNetwork<S: Scalar, R, O> {
    pub reservoir: R,
    pub readout: O,
    _marker: PhantomData<S>,
}

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
