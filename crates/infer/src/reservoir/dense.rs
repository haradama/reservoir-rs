use nalgebra::{DMatrix, DVector};
use reservoir_core::{reservoir::Reservoir, types::*};

#[derive(Debug, Clone)]
pub struct DenseReservoir<S: Scalar> {
    pub w_in: DMatrix<S>,
    pub w: DMatrix<S>,
    pub leaking_rate: S,
    pub input_dim: usize,

    pub res_state: DVector<S>,
    pub ext_state: DVector<S>,
}

impl<S: Scalar> DenseReservoir<S> {
    pub fn create(
        w_in: DMatrix<S>,
        w: DMatrix<S>,
        leaking_rate: S,
        input_dim: usize,
        units: usize,
    ) -> Self {
        Self {
            w_in,
            w,
            leaking_rate,
            input_dim,
            res_state: DVector::zeros(units),
            ext_state: DVector::zeros(1 + input_dim + units),
        }
    }

    fn rebuild_ext_state(&mut self, input: &Input<S>) {
        self.ext_state[0] = S::one();
        self.ext_state.rows_mut(1, self.input_dim).copy_from(input);
        self.ext_state
            .rows_mut(1 + self.input_dim, self.res_state.len())
            .copy_from(&self.res_state);
    }
}

impl<S: Scalar> Reservoir<S> for DenseReservoir<S> {
    fn reset(&mut self) {
        self.res_state.fill(S::zero());
        self.ext_state.fill(S::zero());
    }

    fn step(&mut self, input: &Input<S>) -> &State<S> {
        let pre = &self.w * &self.res_state + &self.w_in * input;
        let act = pre.map(|x| x.activation());

        self.res_state = &self.res_state * (S::one() - self.leaking_rate) + act * self.leaking_rate;

        self.rebuild_ext_state(input);
        &self.ext_state
    }

    fn dim(&self) -> usize {
        self.ext_state.len()
    }

    fn state(&self) -> &State<S> {
        &self.ext_state
    }
}
