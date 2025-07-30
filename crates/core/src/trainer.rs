use crate::{readout::Readout, reservoir::Reservoir, types::Scalar};

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
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
    ) -> Result<(), Self::Error>;
}

pub struct RidgeParams<S: Scalar> {
    pub ridge: S,
}