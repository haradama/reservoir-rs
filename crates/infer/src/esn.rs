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
/// Conversion trait for inputs accepted by [`EchoStateNetwork::predict`].
///
/// This trait exists to make prediction ergonomics better:
/// you can pass `Vec<S>`, `&[S]`, `DVector<S>`, or `heapless::Vec<S, N>`.
///
/// # Notes
/// - Enabled only when the `alloc` feature is active because it produces a `DVector`.
pub trait IntoInput<S: Scalar> {
    /// Convert `self` into a dynamic vector (`nalgebra::DVector`).
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
/// A lightweight wrapper that wires a [`Reservoir`] and a [`Readout`] for inference.
///
/// This type is intentionally minimal:
/// - It does **not** train; you provide the reservoir/readout instances.
/// - It performs a single-step update and then applies the readout.
///
/// # Feature gate
/// Available only with `alloc`, because it operates on dynamic vectors (`DVector`).
///
/// # Example
/// ```rust,ignore
/// use reservoir_infer::EchoStateNetwork;
/// # use reservoir_core::{Reservoir, Readout, Scalar};
/// # // Provide your own Reservoir/Readout implementations.
/// # fn demo<S: Scalar, R: Reservoir<S>, O: Readout<S>>(r: R, o: O) {
/// let mut esn = EchoStateNetwork::<S, _, _>::new(r, o);
/// // let y = esn.predict(vec![0.1, 0.2]);
/// # }
/// ```
pub struct EchoStateNetwork<S: Scalar, R, O> {
    /// Reservoir state transition model.
    pub reservoir: R,
    /// Readout that maps reservoir state to output.
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
    /// Create a new ESN wrapper from a reservoir and a readout.
    pub fn new(reservoir: R, readout: O) -> Self {
        Self {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }

    /// Run one ESN step and produce the readout output.
    ///
    /// This method:
    /// 1. Converts the input into a `DVector` via [`IntoInput`].
    /// 2. Advances the reservoir state (`reservoir.step`).
    /// 3. Applies the readout (`readout.predict`).
    pub fn predict<I>(&mut self, input: I) -> Output<S>
    where
        I: IntoInput<S>,
    {
        let dv = input.into_dvector();
        let state = self.reservoir.step(&dv);
        self.readout.predict(state)
    }

    /// Dimension of the reservoir *extended state* used by the readout.
    ///
    /// For many reservoirs in this crate, the extended state layout is:
    /// `[bias(1), input(input_dim), reservoir_state(units)]`.
    pub fn state_dim(&self) -> usize {
        self.reservoir.dim()
    }
}

/// Static (stack-allocated) reservoir interface.
///
/// This is the `no_std`-friendly counterpart of `reservoir_core::Reservoir`,
/// using fixed-size vectors.
///
/// - `IN`: input dimension
/// - `EXT`: extended-state dimension returned to the readout
pub trait StaticReservoir<S: Scalar, const IN: usize, const EXT: usize> {
    /// Advance the reservoir by one step and return a reference to the extended state.
    fn step_static(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT>;
}

/// Static (stack-allocated) readout interface.
///
/// - `H`: extended-state dimension (readout input)
/// - `OUT`: output dimension
pub trait StaticReadout<S: Scalar, const H: usize, const OUT: usize> {
    /// Map the extended state into an output vector.
    fn predict_static(&self, state: &SVector<S, H>) -> SVector<S, OUT>;
}

/// A simple static ESN wrapper that composes a static reservoir and a static readout.
///
/// This type is designed for `no_std` environments and embedded use-cases.
/// All dimensions are compile-time constants.
pub struct StaticESN<S, R, O> {
    /// Reservoir state transition model (static dimensions).
    pub reservoir: R,
    /// Readout mapping extended state to output (static dimensions).
    pub readout: O,
    _marker: PhantomData<S>,
}

impl<S, R, O> StaticESN<S, R, O>
where
    S: Scalar,
{
    /// Create a new static ESN wrapper.
    pub fn new(reservoir: R, readout: O) -> Self {
        Self {
            reservoir,
            readout,
            _marker: PhantomData,
        }
    }

    /// Run one step and return the output.
    ///
    /// The dimension parameters are explicit so that the compiler can verify
    /// the reservoir/readout compatibility.
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

        let out = esn.predict(vec![1.0, 2.0, 3.0]);
        assert_eq!(out, DVector::from_vec(vec![1.0, 2.0, 3.0]));

        let xs = [4.0, 5.0, 6.0];
        let out = esn.predict(&xs[..]);
        assert_eq!(out, DVector::from_vec(vec![4.0, 5.0, 6.0]));

        let out = esn.predict(DVector::from_vec(vec![7.0, 8.0, 9.0]));
        assert_eq!(out, DVector::from_vec(vec![7.0, 8.0, 9.0]));

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
