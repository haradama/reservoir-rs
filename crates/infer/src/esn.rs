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
impl<S: Scalar> IntoInput<S> for &[S] {
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
        DVector::from_iterator(self.len(), self)
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;
    use heapless::Vec as HVec;
    use nalgebra::DVector;
    use reservoir_core::types::{Input, Output, State};
    use reservoir_core::{Readout, Reservoir};

    #[derive(Debug, Clone)]
    struct PassReservoir<S: Scalar> {
        state: DVector<S>,
    }

    impl<S: Scalar> PassReservoir<S> {
        fn new(dim: usize) -> Self {
            Self {
                state: DVector::zeros(dim),
            }
        }
    }

    impl<S: Scalar> Reservoir<S> for PassReservoir<S> {
        fn reset(&mut self) {
            self.state.fill(S::zero());
        }
        fn step(&mut self, input: &Input<S>) -> &State<S> {
            self.state.copy_from(input);
            &self.state
        }
        fn dim(&self) -> usize {
            self.state.len()
        }
        fn state(&self) -> &State<S> {
            &self.state
        }
    }

    #[derive(Debug, Clone)]
    struct IdentityReadout;

    impl<S: Scalar> Readout<S> for IdentityReadout {
        fn predict(&self, state: &State<S>) -> Output<S> {
            state.clone()
        }
        fn output_dim(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_esn_predict_intoinput_variants() {
        type S = f64;
        let reservoir = PassReservoir::<S>::new(3);
        let readout = IdentityReadout;
        let mut esn = EchoStateNetwork::<S, _, _>::new(reservoir, readout);

        // Vec<S>
        let out = esn.predict(vec![1.0, 2.0, 3.0]);
        assert_eq!(out, DVector::from_vec(vec![1.0, 2.0, 3.0]));

        // &[S]
        let xs = [4.0, 5.0, 6.0];
        let out = esn.predict(&xs[..]);
        assert_eq!(out, DVector::from_vec(vec![4.0, 5.0, 6.0]));

        // DVector<S>
        let out = esn.predict(DVector::from_vec(vec![7.0, 8.0, 9.0]));
        assert_eq!(out, DVector::from_vec(vec![7.0, 8.0, 9.0]));

        // heapless::Vec
        let mut hv: HVec<S, 4> = HVec::new();
        hv.push(10.0).ok();
        hv.push(11.0).ok();
        hv.push(12.0).ok();
        let out = esn.predict(hv);
        assert_eq!(out, DVector::from_vec(vec![10.0, 11.0, 12.0]));
    }

    #[test]
    fn test_esn_state_dim() {
        type S = f64;
        let reservoir = PassReservoir::<S>::new(5);
        let readout = IdentityReadout;
        let esn = EchoStateNetwork::<S, _, _>::new(reservoir, readout);
        assert_eq!(esn.state_dim(), 5);
    }
}
