use nalgebra::{SMatrix, SVector};
use reservoir_core::types::Scalar;

use crate::esn::StaticReservoir as StaticReservoirTrait;

#[derive(Debug, Clone)]
pub struct StaticReservoir<S: Scalar, const IN: usize, const N: usize, const EXT: usize> {
    pub w_in: SMatrix<S, N, IN>,
    pub w: SMatrix<S, N, N>,
    pub leaking_rate: S,
    pub res_state: SVector<S, N>,
    pub ext_state: SVector<S, EXT>,
}

impl<S: Scalar, const IN: usize, const N: usize, const EXT: usize> StaticReservoir<S, IN, N, EXT> {
    pub fn create(w_in: SMatrix<S, N, IN>, w: SMatrix<S, N, N>, leaking_rate: S) -> Self {
        Self {
            w_in,
            w,
            leaking_rate,
            res_state: SVector::zeros(),
            ext_state: SVector::zeros(),
        }
    }

    pub fn step(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT> {
        let pre = self.w * self.res_state + self.w_in * input;
        let act = pre.map(|x| x.activation());
        self.res_state = self.res_state * (S::one() - self.leaking_rate) + act * self.leaking_rate;

        self.ext_state[0] = S::one();
        for i in 0..IN {
            self.ext_state[1 + i] = input[i];
        }
        for i in 0..N {
            self.ext_state[1 + IN + i] = self.res_state[i];
        }

        &self.ext_state
    }
}

impl<S: Scalar, const IN: usize, const N: usize, const EXT: usize> StaticReservoirTrait<S, IN, EXT>
    for StaticReservoir<S, IN, N, EXT>
{
    fn step_static(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT> {
        self.step(input)
    }
}
