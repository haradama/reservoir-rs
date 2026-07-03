use crate::float::RealScalar;
use reservoir_core::types::Scalar;
use reservoir_infer::{EchoStateNetwork, LinearReadout, SparseReservoir};
use std::fmt::Write;

/// Static model code generation utilities.
///
/// This module provides helpers to **export a trained model** (currently: sparse ESN)
/// into **Rust source code** that defines the model weights as `const` arrays.
///
/// The generated code is intended for **embedded / `no_std` inference** use cases,
/// where dynamic allocation and file I/O are undesirable:
///
/// - CSR matrices (`W_in`, `W_res`) are emitted as `u16` index arrays plus `f32` values.
/// - The readout matrix (`W_out`) is emitted as a flattened `f32` array.
/// - The initial reservoir state is emitted as a flattened `f32` array.
///
/// # Output format
/// `generate_sparse_code` returns a single `String` containing Rust code with:
/// - `INPUT_DIM`, `RESERVOIR_SIZE`, `OUTPUT_DIM`, `EXTENDED_SIZE`
/// - `LEAKING_RATE`
/// - CSR metadata (`*_NROWS`, `*_NCOLS`, `*_NNZ`)
/// - CSR arrays: `*_ROW_PTR`, `*_COL_IDX`, `*_VALUES`
/// - `W_OUT_DATA` and `INITIAL_STATE_DATA`
///
/// The generated snippet includes `use` statements for `reservoir-infer` static
/// reservoirs/readouts and `nalgebra` fixed-size types, so it can be pasted into
/// a target crate/module with minimal editing.
///
/// # Limits / invariants
/// - CSR indices are emitted as `u16`. Therefore the number of non-zeros (NNZ)
///   must fit into `u16::MAX` for both `W_in` and `W_res`.
/// - All scalar values are formatted as `f32` with 8 decimal digits.
///   This is a deliberate tradeoff for portability and code size.
/// - The extended state layout assumed here matches `reservoir-infer` reservoirs:
///   `[bias(1), input(input_dim), reservoir_state(reservoir_size)]`.
///
/// # Feature gating
/// This code generator uses `std::fmt::Write` and returns an owned `String`,
/// so it is typically compiled behind the `std` feature of `reservoir-train`.
///
/// # Example
/// ```no_run
/// # use reservoir_train::{ESNBuilder, ESNFitRidge};
/// # use reservoir_train::StaticModelGenerator;
/// // Train a sparse ESN (ridge readout).
/// let mut esn = ESNBuilder::<f32>::new(1, 1)
///     .units(200)
///     .connectivity(8)
///     .input_connectivity(1)
///     .spectral_radius(1.2)
///     .leaking_rate(0.8)
///     .seed(42)
///     .build_sparse();
///
/// // Fit (dummy example; supply your actual data here).
/// // esn.fit(&inputs, &targets, 1e-6, 50);
///
/// // Export as Rust code (paste into your embedded inference crate).
/// let code = StaticModelGenerator::generate_sparse_code(&esn).unwrap();
/// println!("{}", code);
/// ```
pub struct StaticModelGenerator;

impl StaticModelGenerator {
    /// Generate Rust source code for a trained **sparse** Echo State Network.
    ///
    /// This function inspects the provided ESN and serializes its parameters into
    /// Rust `const` definitions suitable for `no_std` inference:
    ///
    /// - `W_in` and `W_res` (reservoir matrices) are emitted in CSR form, using `u16`
    ///   indices (`ROW_PTR`, `COL_IDX`) and `f32` values (`VALUES`).
    /// - The readout weight matrix `W_out` is emitted as a flattened `f32` array.
    /// - The initial reservoir state is emitted as a flattened `f32` array.
    ///
    /// The output includes enough metadata (dimensions / NNZ counts) to validate
    /// the arrays at compile time and to reconstruct the static reservoir/readout
    /// types in a downstream crate.
    ///
    /// # Type parameters
    /// - `S`: training scalar type. Must be convertible/printable (`RealScalar + Display`)
    ///   because values are formatted into source code.
    /// - `O`: readout type. Must implement `reservoir_core::Readout<S>` and [`GetWeights`]
    ///   so this generator can access the dense output weight matrix.
    ///
    /// # Errors
    /// Returns `Err(String)` if the number of non-zeros (NNZ) in `W_in` or `W_res`
    /// exceeds `u16::MAX`, because the generated CSR arrays use `u16` indices.
    ///
    /// # Notes
    /// - All emitted numeric values are formatted as `f32` with 8 decimals.
    /// - This function does not write files; it returns a `String` to give callers
    ///   full control over where the generated code is stored.
    pub fn generate_sparse_code<S, O>(
        esn: &EchoStateNetwork<S, SparseReservoir<S>, O>,
    ) -> Result<String, String>
    where
        S: RealScalar + std::fmt::Display,
        O: reservoir_core::Readout<S> + GetWeights<S>,
    {
        let w_in = &esn.reservoir.w_in;
        let w_res = &esn.reservoir.w;
        let w_out = esn.readout.weights();

        let input_dim = esn.reservoir.input_dim;
        let reservoir_size = esn.reservoir.res_state.len();
        let output_dim = esn.readout.output_dim();
        let ext_size = 1 + input_dim + reservoir_size;
        let leaking_rate = esn.reservoir.leaking_rate;
        let initial_state = &esn.reservoir.res_state;

        if w_in.values.len() > u16::MAX as usize {
            return Err(format!("W_in NNZ ({}) exceeds u16::MAX", w_in.values.len()));
        }
        if w_res.values.len() > u16::MAX as usize {
            return Err(format!(
                "W_res NNZ ({}) exceeds u16::MAX",
                w_res.values.len()
            ));
        }

        let mut code = String::new();

        writeln!(code, "// Auto-generated by reservoir-train::codegen").unwrap();
        writeln!(code, "use reservoir_infer::reservoir::static_sparse_reservoir::{{StaticCsrMatrix, StaticSparseReservoir}};").unwrap();
        writeln!(
            code,
            "use reservoir_infer::readout::static_readout::StaticReadout;"
        )
        .unwrap();
        writeln!(code, "use nalgebra::{{SMatrix, SVector}};").unwrap();
        writeln!(code).unwrap();

        writeln!(code, "pub const INPUT_DIM: usize = {};", input_dim).unwrap();
        writeln!(
            code,
            "pub const RESERVOIR_SIZE: usize = {};",
            reservoir_size
        )
        .unwrap();
        writeln!(code, "pub const OUTPUT_DIM: usize = {};", output_dim).unwrap();
        writeln!(code, "pub const EXTENDED_SIZE: usize = {};", ext_size).unwrap();
        writeln!(code, "pub const LEAKING_RATE: f32 = {:.8};", leaking_rate).unwrap();
        writeln!(code).unwrap();

        writeln!(code, "pub const W_IN_NROWS: usize = {};", w_in.nrows).unwrap();
        writeln!(code, "pub const W_IN_NCOLS: usize = {};", w_in.ncols).unwrap();
        writeln!(code, "pub const W_RES_NROWS: usize = {};", w_res.nrows).unwrap();
        writeln!(code, "pub const W_RES_NCOLS: usize = {};", w_res.ncols).unwrap();
        writeln!(code, "pub const W_IN_NNZ: usize = {};", w_in.values.len()).unwrap();
        writeln!(code, "pub const W_RES_NNZ: usize = {};", w_res.values.len()).unwrap();
        writeln!(code).unwrap();

        let fmt_u16 = |v: &[usize]| -> String {
            v.iter()
                .map(|&x| format!("{}", x as u16))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let fmt_scalar = |v: &[S]| -> String {
            v.iter()
                .map(|x| format!("{:.8}", x))
                .collect::<Vec<_>>()
                .join(", ")
        };

        writeln!(
            code,
            "pub const W_IN_ROW_PTR: [u16; {}] = [{}];",
            w_in.row_ptr.len(),
            fmt_u16(&w_in.row_ptr)
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_IN_COL_IDX: [u16; {}] = [{}];",
            w_in.col_idx.len(),
            fmt_u16(&w_in.col_idx)
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_IN_VALUES: [f32; {}] = [{}];",
            w_in.values.len(),
            fmt_scalar(&w_in.values)
        )
        .unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "pub const W_RES_ROW_PTR: [u16; {}] = [{}];",
            w_res.row_ptr.len(),
            fmt_u16(&w_res.row_ptr)
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_RES_COL_IDX: [u16; {}] = [{}];",
            w_res.col_idx.len(),
            fmt_u16(&w_res.col_idx)
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_RES_VALUES: [f32; {}] = [{}];",
            w_res.values.len(),
            fmt_scalar(&w_res.values)
        )
        .unwrap();
        writeln!(code).unwrap();

        let w_out_flat: Vec<S> = w_out.iter().cloned().collect();
        writeln!(
            code,
            "pub const W_OUT_DATA: [f32; {}] = [{}];",
            w_out_flat.len(),
            fmt_scalar(&w_out_flat)
        )
        .unwrap();
        writeln!(code).unwrap();

        let state_flat: Vec<S> = initial_state.iter().cloned().collect();
        writeln!(
            code,
            "pub const INITIAL_STATE_DATA: [f32; {}] = [{}];",
            state_flat.len(),
            fmt_scalar(&state_flat)
        )
        .unwrap();

        Ok(code)
    }
}

/// Trait for accessing readout weights as a dense matrix.
///
/// The code generator needs a uniform way to retrieve the output weight matrix
/// (`W_out`) from a readout.
///
/// Implemented for [`LinearReadout`] (and therefore for its `RidgeReadout` /
/// `LassoReadout` aliases).
pub trait GetWeights<S: Scalar> {
    /// Borrow the readout weight matrix.
    ///
    /// The matrix shape is `(output_dim, extended_state_dim)` for the readouts in
    /// `reservoir-infer`.
    fn weights(&self) -> &nalgebra::DMatrix<S>;
}

impl<S: Scalar> GetWeights<S> for LinearReadout<S> {
    fn weights(&self) -> &nalgebra::DMatrix<S> {
        &self.w_out
    }
}
