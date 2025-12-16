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
