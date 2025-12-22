use nalgebra::{SMatrix, SVector};
use reservoir_core::types::Scalar;

use crate::esn::StaticReservoir as StaticReservoirTrait;

/// Static (stack-allocated) dense reservoir.
///
/// This is a `no_std`-friendly reservoir implementation using fixed-size
/// matrices and vectors.
///
/// The extended state is laid out as:
/// `[bias(1), input(IN), reservoir_state(N)]`
/// so `EXT` should usually be `1 + IN + N`.
#[derive(Debug, Clone)]
pub struct StaticReservoir<S: Scalar, const IN: usize, const N: usize, const EXT: usize> {
    pub w_in: SMatrix<S, N, IN>,
    pub w: SMatrix<S, N, N>,
    pub leaking_rate: S,
    pub res_state: SVector<S, N>,
    pub ext_state: SVector<S, EXT>,
}

impl<S: Scalar, const IN: usize, const N: usize, const EXT: usize> StaticReservoir<S, IN, N, EXT> {
    /// Create a static reservoir.
    ///
    /// `res_state` and `ext_state` are zero-initialized.
    pub fn create(w_in: SMatrix<S, N, IN>, w: SMatrix<S, N, N>, leaking_rate: S) -> Self {
        Self {
            w_in,
            w,
            leaking_rate,
            res_state: SVector::zeros(),
            ext_state: SVector::zeros(),
        }
    }

    /// Advance one step and return the extended state.
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SMatrix, SVector};

    const EPS: f64 = 1e-12;

    #[test]
    fn test_static_reservoir_step_layout() {
        let w = SMatrix::<f64, 2, 2>::zeros();
        let w_in = SMatrix::<f64, 2, 2>::identity();
        let mut r = StaticReservoir::<f64, 2, 2, 5>::create(w_in, w, 1.0);

        let input = SVector::<f64, 2>::new(0.5, -0.5);
        let s = r.step(&input);

        let a0 = 0.5f64.tanh();
        let a1 = (-0.5f64).tanh();

        assert!((s[0] - 1.0).abs() < EPS);
        assert!((s[1] - 0.5).abs() < EPS);
        assert!((s[2] + 0.5).abs() < EPS);
        assert!((s[3] - a0).abs() < EPS);
        assert!((s[4] - a1).abs() < EPS);
    }
}
