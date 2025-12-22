use nalgebra::SVector;
use reservoir_core::types::Scalar;

use crate::esn::StaticReservoir as StaticReservoirTrait;

/// Static CSR matrix backed by borrowed slices.
///
/// This structure is intended for embedded / `no_std` use-cases where
/// you want to store sparse matrices in ROM/flash and avoid allocation.
///
/// # Panics
/// [`new`] validates CSR invariants and panics if they are violated.
#[derive(Debug, Clone, Copy)]
pub struct StaticCsrMatrix<'a, S: Scalar, const ROWS: usize, const COLS: usize> {
    pub row_ptr: &'a [u16],
    pub col_idx: &'a [u16],
    pub values: &'a [S],
}

impl<'a, S: Scalar, const ROWS: usize, const COLS: usize> StaticCsrMatrix<'a, S, ROWS, COLS> {
    /// Construct a static CSR matrix and validate invariants.
    ///
    /// Invariants:
    /// - `row_ptr.len() == ROWS + 1`
    /// - `col_idx.len() == values.len()`
    /// - `row_ptr[ROWS] == nnz`
    #[inline]
    pub fn new(row_ptr: &'a [u16], col_idx: &'a [u16], values: &'a [S]) -> Self {
        assert_eq!(row_ptr.len(), ROWS + 1, "CSR row_ptr length mismatch");
        assert_eq!(
            col_idx.len(),
            values.len(),
            "CSR col_idx/values length mismatch"
        );
        let nnz = col_idx.len();
        assert_eq!(row_ptr[ROWS] as usize, nnz, "CSR row_ptr[ROWS] != nnz");

        Self {
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Accumulate `y += A * x`.
    ///
    /// Column indices are checked with a `debug_assert` style guard.
    #[inline]
    pub fn matvec_add(&self, x: &SVector<S, COLS>, y: &mut SVector<S, ROWS>) {
        for r in 0..ROWS {
            let start = self.row_ptr[r] as usize;
            let end = self.row_ptr[r + 1] as usize;

            let mut acc = S::zero();
            for k in start..end {
                let c = self.col_idx[k] as usize;
                if c < COLS {
                    acc += self.values[k] * x[c];
                } else {
                    debug_assert!(false, "CSR column index out of bounds");
                }
            }
            y[r] += acc;
        }
    }
}

/// Static sparse reservoir composed from two [`StaticCsrMatrix`] matrices.
///
/// Extended state layout:
/// `[bias(1), input(IN), reservoir_state(N)]`
/// so `EXT` should usually be `1 + IN + N`.
#[derive(Debug, Clone)]
pub struct StaticSparseReservoir<'a, S: Scalar, const IN: usize, const N: usize, const EXT: usize> {
    pub w_in: StaticCsrMatrix<'a, S, N, IN>,
    pub w: StaticCsrMatrix<'a, S, N, N>,
    pub leaking_rate: S,
    pub res_state: SVector<S, N>,
    pub ext_state: SVector<S, EXT>,
}

impl<'a, S: Scalar, const IN: usize, const N: usize, const EXT: usize>
    StaticSparseReservoir<'a, S, IN, N, EXT>
{
    /// Create a static sparse reservoir (states are zero-initialized).
    #[inline]
    pub fn create(
        w_in: StaticCsrMatrix<'a, S, N, IN>,
        w: StaticCsrMatrix<'a, S, N, N>,
        leaking_rate: S,
    ) -> Self {
        Self {
            w_in,
            w,
            leaking_rate,
            res_state: SVector::zeros(),
            ext_state: SVector::zeros(),
        }
    }

    /// Advance one step and return the extended state.
    #[inline]
    pub fn step(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT> {
        let mut pre: SVector<S, N> = SVector::zeros();
        self.w.matvec_add(&self.res_state, &mut pre);
        self.w_in.matvec_add(input, &mut pre);

        for i in 0..N {
            let act = pre[i].activation();
            self.res_state[i] =
                self.res_state[i] * (S::one() - self.leaking_rate) + act * self.leaking_rate;
        }

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

impl<'a, S: Scalar, const IN: usize, const N: usize, const EXT: usize>
    StaticReservoirTrait<S, IN, EXT> for StaticSparseReservoir<'a, S, IN, N, EXT>
{
    #[inline]
    fn step_static(&mut self, input: &SVector<S, IN>) -> &SVector<S, EXT> {
        self.step(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SVector;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_static_csr_new_ok() {
        let row_ptr: [u16; 3] = [0, 1, 2];
        let col_idx: [u16; 2] = [0, 1];
        let values: [f64; 2] = [1.0, 1.0];
        let _ = StaticCsrMatrix::<f64, 2, 2>::new(&row_ptr, &col_idx, &values);
    }

    #[test]
    #[should_panic(expected = "CSR row_ptr length mismatch")]
    fn test_static_csr_new_rowptr_len_mismatch_panics() {
        let row_ptr: [u16; 2] = [0, 0];
        let col_idx: [u16; 0] = [];
        let values: [f64; 0] = [];
        let _ = StaticCsrMatrix::<f64, 2, 2>::new(&row_ptr, &col_idx, &values);
    }

    #[test]
    #[should_panic(expected = "CSR col_idx/values length mismatch")]
    fn test_static_csr_new_col_values_mismatch_panics() {
        let row_ptr: [u16; 3] = [0, 1, 2];
        let col_idx: [u16; 2] = [0, 1];
        let values: [f64; 1] = [1.0];
        let _ = StaticCsrMatrix::<f64, 2, 2>::new(&row_ptr, &col_idx, &values);
    }

    #[test]
    #[should_panic(expected = "CSR row_ptr[ROWS] != nnz")]
    fn test_static_csr_new_nnz_mismatch_panics() {
        let row_ptr: [u16; 3] = [0, 1, 3];
        let col_idx: [u16; 2] = [0, 1];
        let values: [f64; 2] = [1.0, 1.0];
        let _ = StaticCsrMatrix::<f64, 2, 2>::new(&row_ptr, &col_idx, &values);
    }

    #[test]
    fn test_static_sparse_reservoir_step_layout() {
        let w_row_ptr: [u16; 3] = [0, 0, 0];
        let w_col_idx: [u16; 0] = [];
        let w_values: [f64; 0] = [];
        let w = StaticCsrMatrix::<f64, 2, 2>::new(&w_row_ptr, &w_col_idx, &w_values);

        let win_row_ptr: [u16; 3] = [0, 1, 2];
        let win_col_idx: [u16; 2] = [0, 1];
        let win_values: [f64; 2] = [1.0, 1.0];
        let w_in = StaticCsrMatrix::<f64, 2, 2>::new(&win_row_ptr, &win_col_idx, &win_values);

        let mut r = StaticSparseReservoir::<f64, 2, 2, 5>::create(w_in, w, 1.0);

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
