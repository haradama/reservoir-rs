use nalgebra::{DMatrix, DVector};
use reservoir_core::{reservoir::Reservoir, types::*};

/// Dense ESN reservoir using dense matrices.
///
/// The reservoir update is the classic leaky-integrator ESN:
///
/// - `pre = W * r + W_in * u`
/// - `act = activation(pre)`
/// - `r <- (1 - leaking_rate) * r + leaking_rate * act`
///
/// The returned state is an **extended state** vector:
/// `[bias(1), input(input_dim), reservoir_state(units)]`.
///
/// # Notes
/// - `activation()` is provided by the [`Scalar`] implementation in `reservoir-core`.
/// - This is an inference-only component: weight initialization / training is out of scope.
#[derive(Debug, Clone)]
pub struct DenseReservoir<S: Scalar> {
    /// Input weight matrix `W_in` (units x input_dim).
    pub w_in: DMatrix<S>,
    /// Recurrent weight matrix `W` (units x units).
    pub w: DMatrix<S>,
    /// Leaking rate in `[0, 1]` for leaky integration.
    pub leaking_rate: S,
    /// Input dimension used to construct the extended state.
    pub input_dim: usize,

    /// Reservoir state `r` (units).
    pub res_state: DVector<S>,
    /// Extended state `[1, input..., r...]` (1 + input_dim + units).
    pub ext_state: DVector<S>,
}

impl<S: Scalar> DenseReservoir<S> {
    /// Create a dense reservoir.
    ///
    /// `units` determines the internal reservoir dimension.
    /// `ext_state` is initialized with length `1 + input_dim + units`.
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

    /// Rebuild the extended state layout `[bias, input..., res_state...]`.
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    const EPS: f64 = 1e-12;

    #[test]
    fn test_dense_reservoir_step_ext_state_layout_leaking_1() {
        let w = DMatrix::<f64>::zeros(2, 2);
        let w_in = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let mut r = DenseReservoir::create(w_in, w, 1.0, 2, 2);

        let input = DVector::from_vec(vec![0.5, -0.5]);
        let state = r.step(&input);

        let a0 = 0.5f64.tanh();
        let a1 = (-0.5f64).tanh();

        assert_eq!(state.len(), 1 + 2 + 2);
        assert!((state[0] - 1.0).abs() < EPS);
        assert!((state[1] - 0.5).abs() < EPS);
        assert!((state[2] + 0.5).abs() < EPS);
        assert!((state[3] - a0).abs() < EPS);
        assert!((state[4] - a1).abs() < EPS);
    }

    #[test]
    fn test_dense_reservoir_leaking_rate_half() {
        let w = DMatrix::<f64>::zeros(2, 2);
        let w_in = DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let mut r = DenseReservoir::create(w_in, w, 0.5, 2, 2);

        let input = DVector::from_vec(vec![0.5, -0.5]);
        let _ = r.step(&input);

        let a0 = 0.5f64.tanh() * 0.5;
        let a1 = (-0.5f64).tanh() * 0.5;

        assert!((r.res_state[0] - a0).abs() < EPS);
        assert!((r.res_state[1] - a1).abs() < EPS);
    }

    #[test]
    fn test_dense_reservoir_reset() {
        let w = DMatrix::<f64>::identity(2, 2);
        let w_in = DMatrix::<f64>::zeros(2, 2);
        let mut r = DenseReservoir::create(w_in, w, 0.5, 2, 2);

        let input = DVector::from_vec(vec![1.0, 1.0]);
        let _ = r.step(&input);

        r.reset();
        assert!(r.res_state.iter().all(|&v| v == 0.0));
        assert!(r.ext_state.iter().all(|&v| v == 0.0));
    }
}
