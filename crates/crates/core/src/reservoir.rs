use crate::types::{Input, Scalar, State};

pub trait Reservoir<S: Scalar> {
    fn reset(&mut self);
    fn step(&mut self, input: &Input<S>) -> &State<S>;
    fn dim(&self) -> usize;
    fn state(&self) -> &State<S>;
}
