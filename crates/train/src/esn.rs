use crate::RngType;
use crate::{
    float::RealScalar,
    trainer::{LassoTrainer, RidgeTrainer},
};
use alloc::vec::Vec;
use core::marker::PhantomData;
use nalgebra::{DMatrix, DVector, Normed};
use rand::RngCore;
use rand::{distributions::Uniform, Rng, SeedableRng};
use reservoir_core::{types::*, Input, Output, Reservoir, Trainer};
use reservoir_infer::reservoir::CsrMatrix;
use reservoir_infer::{
    DenseReservoir, EchoStateNetwork, LassoReadout, RidgeReadout, SparseReservoir,
};

/// Echo State Network (ESN) training helpers and model builders.
///
/// This module provides two layers of functionality:
///
/// 1. **Convenience traits** ([`ESNFitRidge`], [`ESNFitLasso`]) to train common ESN
///    combinations with a single method call using `Vec<Vec<S>>` inputs/targets.
/// 2. A configurable **builder** ([`ESNBuilder`]) to construct randomly initialized ESNs
///    (dense or sparse reservoirs, ridge or lasso readouts), with deterministic seeding.
///
/// The runtime reservoir / readout implementations are provided by `reservoir-infer`,
/// while the optimization logic (ridge / lasso solvers) is implemented in `crate::trainer`.
///
/// # Feature and environment notes
/// - This crate is `no_std`-friendly, but relies on `alloc` for training-time data structures.
/// - Random generation uses `rand` and the crate-defined [`RngType`], which maps to:
///   - `StdRng` under the `std` feature
///   - `SmallRng` under `no_std`
///
/// # Design choices
/// - The builder generates *untrained* reservoirs/readouts. Training is performed via
///   [`ESNFitRidge`] / [`ESNFitLasso`] or by directly using the lower-level [`Trainer`] API.
/// - The dense reservoir uses eigenvalue computation to scale to a target spectral radius.
/// - The sparse reservoir uses power iteration to estimate the spectral radius cheaply.
///
/// # Examples
/// Ridge training with dense reservoir:
/// ```no_run
/// # use reservoir_train::{ESNBuilder, ESNFitRidge};
/// let mut esn = ESNBuilder::<f32>::new(1, 1)
///     .units(200)
///     .spectral_radius(1.2)
///     .leaking_rate(0.8)
///     .input_scaling(1.0)
///     .seed(42)
///     .build();
///
/// // inputs/targets are `Vec<Vec<f32>>`
/// // esn.fit(&inputs, &targets, 1e-6, 50);
/// ```
///
/// Lasso training with sparse reservoir:
/// ```no_run
/// # use reservoir_train::{ESNBuilder, ESNFitLasso};
/// let mut esn = ESNBuilder::<f32>::new(1, 1)
///     .units(300)
///     .connectivity(8)
///     .input_connectivity(1)
///     .spectral_radius(0.95)
///     .leaking_rate(0.3)
///     .seed(7)
///     .build_sparse_lasso();
///
/// // esn.fit_lasso(&inputs, &targets, 1e-4, 1000, 1e-6, 50);
/// ```
pub trait ESNFitRidge<S>
where
    S: RealScalar,
{
    /// Train the ESN readout weights using ridge regression.
    ///
    /// This is a convenience wrapper around [`RidgeTrainer`]. Inputs and targets
    /// are provided as `Vec<Vec<S>>` and are internally converted to `nalgebra::DVector`
    /// (`reservoir_core::types::Input` / `Output`).
    ///
    /// - `ridge`: L2 regularization strength (added to the diagonal of `X^T X`).
    /// - `washout`: number of initial time steps to ignore (state warm-up).
    ///
    /// # Panics
    /// This method panics if training fails. If you need fallible behavior, call the
    /// underlying trainer directly through the [`Trainer`] trait.
    fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize);
}

impl<S> ESNFitRidge<S> for EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>>
where
    S: RealScalar,
{
    fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("training failed");
    }
}

impl<S> ESNFitRidge<S> for EchoStateNetwork<S, SparseReservoir<S>, RidgeReadout<S>>
where
    S: RealScalar,
{
    fn fit(&mut self, inputs: &[Vec<S>], targets: &[Vec<S>], ridge: S, washout: usize) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = RidgeTrainer { ridge };
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("training failed");
    }
}

/// Convenience training API for ESNs using LASSO (L1) regression.
///
/// This trait mirrors [`ESNFitRidge`], but uses coordinate-descent LASSO training
/// implemented by [`LassoTrainer`].
pub trait ESNFitLasso<S>
where
    S: Scalar,
{
    /// Train the ESN readout weights using LASSO regression.
    ///
    /// - `alpha`: L1 regularization strength (sparsity control).
    /// - `max_iter`: maximum coordinate-descent iterations per output dimension.
    /// - `tol`: convergence threshold (max coefficient change).
    /// - `washout`: number of initial steps to ignore.
    ///
    /// # Panics
    /// This method panics if training fails. If you need fallible behavior, call the
    /// underlying trainer directly through the [`Trainer`] trait.
    fn fit_lasso(
        &mut self,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
        alpha: S,
        max_iter: usize,
        tol: S,
        washout: usize,
    );
}

impl<S> ESNFitLasso<S> for EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>>
where
    S: Scalar,
{
    fn fit_lasso(
        &mut self,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
        alpha: S,
        max_iter: usize,
        tol: S,
        washout: usize,
    ) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = LassoTrainer::new(alpha, max_iter, tol);
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("lasso training failed");
    }
}

impl<S> ESNFitLasso<S> for EchoStateNetwork<S, SparseReservoir<S>, LassoReadout<S>>
where
    S: Scalar,
{
    fn fit_lasso(
        &mut self,
        inputs: &[Vec<S>],
        targets: &[Vec<S>],
        alpha: S,
        max_iter: usize,
        tol: S,
        washout: usize,
    ) {
        let inputs_dv: Vec<Input<S>> = inputs.iter().cloned().map(DVector::from_vec).collect();
        let targets_dv: Vec<Output<S>> = targets.iter().cloned().map(DVector::from_vec).collect();

        let mut trainer = LassoTrainer::new(alpha, max_iter, tol);
        trainer
            .fit(
                &mut self.reservoir,
                &mut self.readout,
                &inputs_dv,
                &targets_dv,
                washout,
            )
            .expect("lasso training failed");
    }
}

/// Builder for randomly initialized Echo State Networks.
///
/// [`ESNBuilder`] produces an [`EchoStateNetwork`] with:
/// - a **dense** or **sparse** reservoir, and
/// - a **ridge** or **lasso** readout (initialized randomly).
///
/// The builder is deterministic with respect to [`seed`](ESNBuilder::seed).
///
/// # Parameters
/// - `input_dim`, `output_dim`: external I/O dimensions.
/// - `units`: number of reservoir neurons.
/// - `spectral_radius`: target spectral radius of the reservoir recurrence.
/// - `input_scaling`: scaling applied to input weights.
/// - `leaking_rate`: leaky integrator coefficient (1.0 = no leak / fully new state).
/// - `connectivity`: (sparse) non-zeros per row in recurrent weight CSR matrix.
/// - `input_connectivity`: (sparse) non-zeros per row in input weight CSR matrix.
///
/// # Reservoir scaling
/// - Dense: computes eigenvalues (via `nalgebra`) and rescales the matrix exactly.
/// - Sparse: estimates spectral radius via power iteration and rescales approximately.
///
/// # Output dimension of the reservoir
/// `reservoir-infer` reservoirs expose an *extended state* with the layout:
/// `1 (bias) + input_dim + units`.
/// Readout weights are initialized to map this extended state to `output_dim`.
pub struct ESNBuilder<S: Scalar> {
    input_dim: usize,
    output_dim: usize,
    units: usize,
    spectral_radius: S,
    input_scaling: S,
    leaking_rate: S,
    seed: u64,

    connectivity: usize,
    input_connectivity: usize,

    _marker: PhantomData<S>,
}

impl<S: Scalar> ESNBuilder<S> {
    /// Create a new builder with default hyperparameters.
    ///
    /// Defaults:
    /// - `units = 100`
    /// - `spectral_radius = 1`
    /// - `input_scaling = 1`
    /// - `leaking_rate = 1`
    /// - `seed = 42`
    /// - `connectivity = 8` (sparse recurrent)
    /// - `input_connectivity = max(input_dim, 1)` (sparse input)
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            units: 100,
            spectral_radius: S::one(),
            input_scaling: S::one(),
            leaking_rate: S::one(),
            seed: 42,
            connectivity: 8,
            input_connectivity: input_dim.max(1),
            _marker: PhantomData,
        }
    }

    /// Set the number of reservoir units.
    pub fn units(mut self, n: usize) -> Self {
        self.units = n;
        self
    }

    /// Set the target spectral radius of the recurrent weight matrix.
    pub fn spectral_radius(mut self, r: S) -> Self {
        self.spectral_radius = r;
        self
    }

    /// Set the scaling applied to input weights.
    pub fn input_scaling(mut self, s: S) -> Self {
        self.input_scaling = s;
        self
    }

    /// Set the leaking rate `α` used by the reservoir update.
    ///
    /// Typical range is `(0, 1]`.
    pub fn leaking_rate(mut self, a: S) -> Self {
        self.leaking_rate = a;
        self
    }

    /// Set the RNG seed used for all random initialization.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set sparse recurrent connectivity: number of non-zeros per row in `W_res`.
    ///
    /// The provided value is clamped to at least 1.
    pub fn connectivity(mut self, k: usize) -> Self {
        self.connectivity = k.max(1);
        self
    }

    /// Set sparse input connectivity: number of non-zeros per row in `W_in`.
    ///
    /// The provided value is clamped to at least 1.
    pub fn input_connectivity(mut self, k: usize) -> Self {
        self.input_connectivity = k.max(1);
        self
    }

    /// Generate dense reservoir matrices `(W_in, W_res)` and scale `W_res` to the
    /// desired spectral radius.
    ///
    /// - Random entries are sampled uniformly from `[-0.5, 0.5)`.
    /// - Spectral radius is computed from eigenvalues of the `f64`-mapped matrix.
    /// - If the current radius is non-zero, the matrix is rescaled linearly.
    fn generate_reservoir_matrices(&self) -> (DMatrix<S>, DMatrix<S>) {
        let mut rng = RngType::seed_from_u64(self.seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);
        let mut rnd =
            |r: usize, c: usize| DMatrix::from_fn(r, c, |_, _| S::from_f64_val(rng.sample(uni)));

        let mut w = rnd(self.units, self.units);

        let w_f64 = w.map(|x| x.to_f64_val());
        let eigenvalues = w_f64.complex_eigenvalues();

        let current_rho = eigenvalues
            .iter()
            .map(|c| c.norm())
            .fold(0.0f64, |a, b| a.max(b));

        let target_rho = self.spectral_radius.to_f64_val();

        if current_rho > 0.0 {
            let scale = target_rho / current_rho;
            w *= S::from_f64_val(scale);
        }

        let w_in = rnd(self.units, self.input_dim) * self.input_scaling;

        (w_in, w)
    }

    /// Generate a random dense readout weight matrix.
    ///
    /// Values are sampled uniformly from `[-0.5, 0.5)`.
    fn generate_readout_matrix(&self, input_dim: usize, output_dim: usize) -> DMatrix<S> {
        let mut rng = RngType::seed_from_u64(self.seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);
        DMatrix::from_fn(output_dim, input_dim, |_, _| {
            S::from_f64_val(rng.sample(uni))
        })
    }

    /// Generate a random CSR matrix with exactly `k_per_row` unique columns per row.
    ///
    /// - `k_per_row` is clamped to `[1, cols]`.
    /// - Column indices in each row are sorted increasingly.
    /// - Values are sampled uniformly from `[-0.5, 0.5)`.
    fn gen_sparse_csr(&self, rows: usize, cols: usize, k_per_row: usize) -> CsrMatrix<S> {
        let mut rng = RngType::seed_from_u64(self.seed);
        let uni = Uniform::new(-0.5f64, 0.5f64);

        let k = k_per_row.min(cols).max(1);

        let mut row_ptr: Vec<usize> = Vec::with_capacity(rows + 1);
        let mut col_idx: Vec<usize> = Vec::with_capacity(rows * k);
        let mut values: Vec<S> = Vec::with_capacity(rows * k);

        row_ptr.push(0);

        for _r in 0..rows {
            let mut chosen: Vec<usize> = Vec::with_capacity(k);
            while chosen.len() < k {
                let c = (rng.next_u32() as usize) % cols;
                if !chosen.contains(&c) {
                    chosen.push(c);
                }
            }
            chosen.sort_unstable();

            for &c in &chosen {
                col_idx.push(c);
                values.push(S::from_f64_val(rng.sample(uni)));
            }
            row_ptr.push(col_idx.len());
        }

        CsrMatrix {
            nrows: rows,
            ncols: cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Estimate the spectral radius of a CSR matrix using power iteration.
    ///
    /// This computes an approximation of `|λ_max|` without forming a dense matrix.
    /// The returned value is non-negative.
    fn estimate_rho_power_iter(&self, w: &CsrMatrix<S>, iters: usize) -> f64 {
        let n = w.nrows;
        let mut v: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) / (n as f64)).collect();

        for _ in 0..iters {
            let mut y = vec![0.0f64; n];
            for (r, y_r) in y.iter_mut().enumerate() {
                let start = w.row_ptr[r];
                let end = w.row_ptr[r + 1];
                let mut acc = 0.0;
                for k in start..end {
                    let c = w.col_idx[k];
                    acc += w.values[k].to_f64_val() * v[c];
                }
                *y_r = acc;
            }

            let norm = y.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                for i in 0..n {
                    v[i] = y[i] / norm;
                }
            }
        }

        let mut wv = vec![0.0f64; n];
        for (r, wv_r) in wv.iter_mut().enumerate() {
            let start = w.row_ptr[r];
            let end = w.row_ptr[r + 1];
            let mut acc = 0.0;
            for k in start..end {
                let c = w.col_idx[k];
                acc += w.values[k].to_f64_val() * v[c];
            }
            *wv_r = acc;
        }

        let num = v.iter().zip(wv.iter()).map(|(a, b)| a * b).sum::<f64>();
        let den = v.iter().map(|a| a * a).sum::<f64>();
        (num / den).abs()
    }

    /// Scale all values in a CSR matrix by a scalar factor.
    fn scale_csr(&self, w: &mut CsrMatrix<S>, scale: f64) {
        let s = S::from_f64_val(scale);
        for v in w.values.iter_mut() {
            *v *= s;
        }
    }

    /// Build a **dense reservoir + ridge readout** ESN.
    ///
    /// - Initializes `W_res` and rescales it to the configured spectral radius.
    /// - Initializes `W_in` and applies input scaling.
    /// - Initializes a random ridge readout matrix with shape
    ///   `(output_dim, 1 + input_dim + units)`.
    pub fn build(self) -> EchoStateNetwork<S, DenseReservoir<S>, RidgeReadout<S>> {
        let (w_in, w) = self.generate_reservoir_matrices();

        let reservoir =
            DenseReservoir::create(w_in, w, self.leaking_rate, self.input_dim, self.units);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);

        let readout = RidgeReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }

    /// Build a **dense reservoir + lasso readout** ESN.
    ///
    /// This is identical to [`build`](ESNBuilder::build) except that the readout is
    /// [`LassoReadout`] (weights are still initialized randomly; sparsity is induced
    /// during training).
    pub fn build_lasso(self) -> EchoStateNetwork<S, DenseReservoir<S>, LassoReadout<S>> {
        let (w_in, w) = self.generate_reservoir_matrices();

        let reservoir =
            DenseReservoir::create(w_in, w, self.leaking_rate, self.input_dim, self.units);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);

        let readout = LassoReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }

    /// Build a **sparse reservoir + ridge readout** ESN.
    ///
    /// - Generates CSR `W_res` with `connectivity` non-zeros per row and rescales it
    ///   to match the configured spectral radius (power-iteration estimate).
    /// - Generates CSR `W_in` with `input_connectivity` non-zeros per row and scales it
    ///   by `input_scaling`.
    /// - Initializes a random ridge readout matrix with shape
    ///   `(output_dim, 1 + input_dim + units)`.
    pub fn build_sparse(self) -> EchoStateNetwork<S, SparseReservoir<S>, RidgeReadout<S>> {
        let mut w = self.gen_sparse_csr(self.units, self.units, self.connectivity);

        let rho = self.estimate_rho_power_iter(&w, 50);
        let target = self.spectral_radius.to_f64_val();
        if rho > 0.0 {
            self.scale_csr(&mut w, target / rho);
        }

        let mut w_in = self.gen_sparse_csr(self.units, self.input_dim, self.input_connectivity);

        let s_in = self.input_scaling.to_f64_val();
        if s_in != 1.0 {
            self.scale_csr(&mut w_in, s_in);
        }

        let reservoir = SparseReservoir::create(w_in, w, self.leaking_rate, self.input_dim);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);
        let readout = RidgeReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }

    /// Build a **sparse reservoir + lasso readout** ESN.
    ///
    /// This is identical to [`build_sparse`](ESNBuilder::build_sparse) except that the
    /// readout is [`LassoReadout`].
    pub fn build_sparse_lasso(self) -> EchoStateNetwork<S, SparseReservoir<S>, LassoReadout<S>> {
        let mut w = self.gen_sparse_csr(self.units, self.units, self.connectivity);

        let rho = self.estimate_rho_power_iter(&w, 50);
        let target = self.spectral_radius.to_f64_val();
        if rho > 0.0 {
            self.scale_csr(&mut w, target / rho);
        }

        let mut w_in = self.gen_sparse_csr(self.units, self.input_dim, self.input_connectivity);

        let s_in = self.input_scaling.to_f64_val();
        if s_in != 1.0 {
            self.scale_csr(&mut w_in, s_in);
        }

        let reservoir = SparseReservoir::create(w_in, w, self.leaking_rate, self.input_dim);

        let reservoir_output_dim = reservoir.dim();
        let w_out = self.generate_readout_matrix(reservoir_output_dim, self.output_dim);
        let readout = LassoReadout::create(w_out);

        EchoStateNetwork::new(reservoir, readout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Normed;

    #[test]
    fn test_esn_builder_dense_shapes_and_ext_dim() {
        let esn = ESNBuilder::<f64>::new(3, 2)
            .units(10)
            .spectral_radius(0.9)
            .leaking_rate(0.3)
            .seed(123)
            .build();

        assert_eq!(esn.reservoir.input_dim, 3);
        assert_eq!(esn.reservoir.res_state.len(), 10);
        assert_eq!(esn.reservoir.ext_state.len(), 1 + 3 + 10);
        assert_eq!(esn.readout.output_dim, 2);

        assert_eq!(esn.reservoir.w.nrows(), 10);
        assert_eq!(esn.reservoir.w.ncols(), 10);
        assert_eq!(esn.reservoir.w_in.nrows(), 10);
        assert_eq!(esn.reservoir.w_in.ncols(), 3);
    }

    #[test]
    fn test_esn_builder_dense_spectral_radius_close() {
        let target = 1.2f64;
        let esn = ESNBuilder::<f64>::new(2, 1)
            .units(20)
            .spectral_radius(target)
            .seed(7)
            .build();

        let w_f64 = esn.reservoir.w.map(|x| x.to_f64_val());
        let ev = w_f64.complex_eigenvalues();
        let rho = ev.iter().map(|c| c.norm()).fold(0.0f64, |a, b| a.max(b));

        assert!((rho - target).abs() < 1e-6);
    }

    #[test]
    fn test_esn_builder_sparse_shapes_and_basic_invariants() {
        let esn = ESNBuilder::<f64>::new(4, 1)
            .units(30)
            .connectivity(3)
            .input_connectivity(2)
            .spectral_radius(0.95)
            .seed(42)
            .build_sparse();

        assert_eq!(esn.reservoir.input_dim, 4);
        assert_eq!(esn.reservoir.res_state.len(), 30);
        assert_eq!(esn.reservoir.ext_state.len(), 1 + 4 + 30);

        let w = &esn.reservoir.w;
        assert_eq!(w.row_ptr.len(), w.nrows + 1);
        assert_eq!(*w.row_ptr.last().unwrap(), w.values.len());
        assert_eq!(w.col_idx.len(), w.values.len());

        let w_in = &esn.reservoir.w_in;
        assert_eq!(w_in.row_ptr.len(), w_in.nrows + 1);
        assert_eq!(*w_in.row_ptr.last().unwrap(), w_in.values.len());
        assert_eq!(w_in.col_idx.len(), w_in.values.len());
    }
}
