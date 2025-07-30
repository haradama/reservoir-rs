use crate::types::{Output, Scalar, State};

pub trait Readout<S: Scalar> {
    fn predict(&self, state: &State<S>) -> Output<S>;
    fn output_dim(&self) -> usize;
}
