#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use nalgebra::DVector;

use reservoir_core::{reservoir::Reservoir, types::*};

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct CsrMatrix<S: Scalar> {
    pub nrows: usize,
    pub ncols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<S>,
}

#[cfg(feature = "alloc")]
impl<S: Scalar> CsrMatrix<S> {
    #[inline]
    pub fn matvec(&self, x: &DVector<S>, y: &mut DVector<S>) {
        debug_assert_eq!(x.len(), self.ncols);
        debug_assert_eq!(y.len(), self.nrows);

        y.fill(S::zero());
        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut acc = S::zero();
            for k in start..end {
                let c = self.col_idx[k];
                acc += self.values[k] * x[c];
            }
            y[r] = acc;
        }
    }

    #[inline]
    pub fn matvec_add(&self, x: &DVector<S>, y: &mut DVector<S>) {
        debug_assert_eq!(x.len(), self.ncols);
        debug_assert_eq!(y.len(), self.nrows);

        for r in 0..self.nrows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            let mut acc = S::zero();
            for k in start..end {
                let c = self.col_idx[k];
                acc += self.values[k] * x[c];
            }
            y[r] += acc;
        }
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SparseReservoir<S: Scalar> {
    pub w_in: CsrMatrix<S>,
    pub w: CsrMatrix<S>,
    pub leaking_rate: S,
    pub input_dim: usize,

    pub res_state: DVector<S>,
    pub ext_state: DVector<S>,

    pre: DVector<S>,
}

#[cfg(feature = "alloc")]
impl<S: Scalar> SparseReservoir<S> {
    pub fn create(w_in: CsrMatrix<S>, w: CsrMatrix<S>, leaking_rate: S, input_dim: usize) -> Self {
        let units = w.nrows;
        Self {
            w_in,
            w,
            leaking_rate,
            input_dim,
            res_state: DVector::zeros(units),
            ext_state: DVector::zeros(1 + input_dim + units),
            pre: DVector::zeros(units),
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

#[cfg(feature = "alloc")]
impl<S: Scalar> Reservoir<S> for SparseReservoir<S> {
    fn reset(&mut self) {
        self.res_state.fill(S::zero());
        self.ext_state.fill(S::zero());
        self.pre.fill(S::zero());
    }

    fn step(&mut self, input: &Input<S>) -> &State<S> {
        self.pre.fill(S::zero());
        self.w.matvec_add(&self.res_state, &mut self.pre);
        self.w_in.matvec_add(input, &mut self.pre);

        for i in 0..self.res_state.len() {
            let act = self.pre[i].activation();
            self.res_state[i] =
                self.res_state[i] * (S::one() - self.leaking_rate) + act * self.leaking_rate;
        }

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
    use nalgebra::DVector;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_csr_matvec_and_matvec_add() {
        // 2x2 Identity
        let csr = CsrMatrix::<f64> {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_idx: vec![0, 1],
            values: vec![1.0, 1.0],
        };

        let x = DVector::from_vec(vec![2.0, 3.0]);
        let mut y = DVector::zeros(2);

        csr.matvec(&x, &mut y);
        assert!((y[0] - 2.0).abs() < EPS);
        assert!((y[1] - 3.0).abs() < EPS);

        let mut y2 = DVector::from_vec(vec![1.0, 1.0]);
        csr.matvec_add(&x, &mut y2);
        assert!((y2[0] - 3.0).abs() < EPS);
        assert!((y2[1] - 4.0).abs() < EPS);
    }

    #[test]
    fn test_sparse_reservoir_step_ext_state_layout_leaking_1() {
        // units=2, input_dim=2
        // w: zero
        let w = CsrMatrix::<f64> {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 0, 0],
            col_idx: vec![],
            values: vec![],
        };
        // w_in: identity (2x2)
        let w_in = CsrMatrix::<f64> {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_idx: vec![0, 1],
            values: vec![1.0, 1.0],
        };

        let mut r = SparseReservoir::create(w_in, w, 1.0, 2);
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
    fn test_sparse_reservoir_reset() {
        let w = CsrMatrix::<f64> {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 0, 0],
            col_idx: vec![],
            values: vec![],
        };
        let w_in = CsrMatrix::<f64> {
            nrows: 2,
            ncols: 2,
            row_ptr: vec![0, 1, 2],
            col_idx: vec![0, 1],
            values: vec![1.0, 1.0],
        };

        let mut r = SparseReservoir::create(w_in, w, 0.5, 2);
        let input = DVector::from_vec(vec![1.0, 1.0]);
        let _ = r.step(&input);

        r.reset();
        assert!(r.res_state.iter().all(|&v| v == 0.0));
        assert!(r.ext_state.iter().all(|&v| v == 0.0));
        assert!(r.pre.iter().all(|&v| v == 0.0));
    }
}
