use nalgebra::SVector;
use reservoir_core::types::Scalar;

use crate::esn::StaticReservoir as StaticReservoirTrait;

#[derive(Debug, Clone, Copy)]
pub struct StaticCsrMatrix<'a, S: Scalar, const ROWS: usize, const COLS: usize> {
    pub row_ptr: &'a [u16],
    pub col_idx: &'a [u16],
    pub values: &'a [S],
}

impl<'a, S: Scalar, const ROWS: usize, const COLS: usize> StaticCsrMatrix<'a, S, ROWS, COLS> {
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
