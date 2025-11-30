#![cfg_attr(not(feature = "std"), allow(unused_imports))]

use crate::{
    readout::Readout,
    reservoir::Reservoir,
    types::{Input, Output, Scalar},
};

pub trait Trainer<R, O, S>
where
    R: Reservoir<S>,
    O: Readout<S>,
    S: Scalar,
{
    type Error;

    fn fit(
        &mut self,
        reservoir: &mut R,
        readout: &mut O,
        inputs: &[Input<S>],
        targets: &[Output<S>],
        washout: usize,
    ) -> Result<(), Self::Error>;
}

pub struct RidgeParams<S: Scalar> {
    pub ridge: S,
}
