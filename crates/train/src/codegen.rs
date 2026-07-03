use crate::float::RealScalar;
use reservoir_core::types::Scalar;
use reservoir_infer::{DenseReservoir, EchoStateNetwork, LinearReadout, SparseReservoir};
use std::fmt::Write;

/// Static model code generation utilities.
///
/// This module provides helpers to **export a trained model** into **Rust source
/// code** that defines the model weights as `const` arrays. There are two entry
/// points, matching the two reservoir kinds:
///
/// - [`generate_dense_code`](StaticModelGenerator::generate_dense_code) for a
///   trained [`DenseReservoir`] ESN.
/// - [`generate_sparse_code`](StaticModelGenerator::generate_sparse_code) for a
///   trained [`SparseReservoir`] ESN (CSR).
///
/// This is the *train → infer-only* hand-off: the generated module depends only
/// on `reservoir-infer` (+ `nalgebra`) and reconstructs the model with the static,
/// `no_std`-friendly `StaticReservoir` / `StaticReadout` / `StaticESN` types. Once
/// generated, the inference build no longer needs `reservoir-train`.
///
/// The generated code is intended for **embedded / `no_std` inference** use cases,
/// where dynamic allocation and file I/O are undesirable:
///
/// - Dense matrices (`W_in`, `W`) are emitted as flattened column-major `f32` arrays.
/// - CSR matrices (`W_in`, `W_res`) are emitted as `u16` index arrays plus `f32` values.
/// - The readout matrix (`W_out`) is emitted as a flattened column-major `f32` array.
/// - The initial reservoir state is emitted as a flattened `f32` array.
///
/// Both generators emit dense `const` matrices (the readout, and — for the dense
/// reservoir — `W_in` / `W`) in **column-major** order so they can be rebuilt with
/// `nalgebra`'s `from_column_slice`. Both also emit ready-to-use `build()` / `step()`
/// helpers and `Reservoir` / `Readout` / `Model` type aliases, so the consumer does
/// not have to wire the const-generic dimensions by hand.
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
    /// - The module also emits `Reservoir` / `Readout` / `Model` type aliases and
    ///   ready-to-use `build()` / `step()` helpers, so the inference crate can call
    ///   `model::build()` without wiring the const-generic dimensions by hand.
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

        writeln!(
            code,
            "// Auto-generated by reservoir-train::codegen (sparse)"
        )
        .unwrap();
        // Not every emitted const is used by `build()` (CSR metadata is kept for
        // callers that wire the model by hand), so silence dead-code in consumers.
        writeln!(code, "#![allow(dead_code)]").unwrap();
        writeln!(code, "use reservoir_infer::reservoir::static_sparse_reservoir::{{StaticCsrMatrix, StaticSparseReservoir}};").unwrap();
        writeln!(
            code,
            "use reservoir_infer::readout::static_readout::StaticReadout;"
        )
        .unwrap();
        writeln!(code, "use reservoir_infer::esn::StaticESN;").unwrap();
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
        writeln!(code).unwrap();

        writeln!(
            code,
            "pub type Reservoir = StaticSparseReservoir<'static, f32, INPUT_DIM, RESERVOIR_SIZE, EXTENDED_SIZE>;"
        )
        .unwrap();
        writeln!(
            code,
            "pub type Readout = StaticReadout<f32, EXTENDED_SIZE, OUTPUT_DIM>;"
        )
        .unwrap();
        writeln!(code, "pub type Model = StaticESN<f32, Reservoir, Readout>;").unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "/// Reconstruct the trained model using only `reservoir-infer`."
        )
        .unwrap();
        writeln!(code, "pub fn build() -> Model {{").unwrap();
        writeln!(
            code,
            "    let w_in = StaticCsrMatrix::<f32, RESERVOIR_SIZE, INPUT_DIM>::new(&W_IN_ROW_PTR, &W_IN_COL_IDX, &W_IN_VALUES);"
        )
        .unwrap();
        writeln!(
            code,
            "    let w_res = StaticCsrMatrix::<f32, RESERVOIR_SIZE, RESERVOIR_SIZE>::new(&W_RES_ROW_PTR, &W_RES_COL_IDX, &W_RES_VALUES);"
        )
        .unwrap();
        writeln!(
            code,
            "    let mut reservoir = Reservoir::create(w_in, w_res, LEAKING_RATE);"
        )
        .unwrap();
        writeln!(
            code,
            "    reservoir.res_state = SVector::<f32, RESERVOIR_SIZE>::from_column_slice(&INITIAL_STATE_DATA);"
        )
        .unwrap();
        writeln!(
            code,
            "    let w_out = SMatrix::<f32, OUTPUT_DIM, EXTENDED_SIZE>::from_column_slice(&W_OUT_DATA);"
        )
        .unwrap();
        writeln!(code, "    let readout = Readout::create(w_out);").unwrap();
        writeln!(code, "    StaticESN::new(reservoir, readout)").unwrap();
        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "/// Advance the model by one step (hides the const-generic dimensions)."
        )
        .unwrap();
        writeln!(
            code,
            "pub fn step(model: &mut Model, x: &SVector<f32, INPUT_DIM>) -> SVector<f32, OUTPUT_DIM> {{"
        )
        .unwrap();
        writeln!(
            code,
            "    model.predict::<INPUT_DIM, OUTPUT_DIM, EXTENDED_SIZE>(x)"
        )
        .unwrap();
        writeln!(code, "}}").unwrap();

        Ok(code)
    }

    /// Generate Rust source code for a trained **dense** Echo State Network.
    ///
    /// The output is a self-contained module that depends only on `reservoir-infer`
    /// and `nalgebra`, reconstructing the model with the static (`no_std`-friendly)
    /// [`StaticReservoir`](reservoir_infer::StaticReservoir) /
    /// [`StaticReadout`](reservoir_infer::StaticReadout) /
    /// [`StaticESN`](reservoir_infer::StaticESN) types. Once generated, the inference
    /// build no longer needs `reservoir-train`.
    ///
    /// Emitted items:
    /// - `INPUT_DIM`, `RESERVOIR_SIZE`, `OUTPUT_DIM`, `EXTENDED_SIZE`, `LEAKING_RATE`
    /// - `W_IN_DATA`, `W_DATA`, `W_OUT_DATA`, `INITIAL_STATE_DATA` (column-major `f32`)
    /// - `Reservoir` / `Readout` / `Model` type aliases with the concrete dimensions
    /// - `build() -> Model` and `step(&mut Model, &SVector) -> SVector` helpers
    ///
    /// # Type parameters
    /// - `S`: training scalar type (`RealScalar + Display`; values are formatted as `f32`).
    /// - `O`: readout type implementing [`GetWeights`].
    ///
    /// # Errors
    /// Never returns `Err` today (the signature mirrors [`generate_sparse_code`] and
    /// leaves room for future validation), but the `Result` should still be handled.
    ///
    /// # Notes
    /// Matrices are emitted in **column-major** order and rebuilt via
    /// `nalgebra::from_column_slice`, matching the storage order of the dynamic
    /// [`DenseReservoir`] they came from.
    pub fn generate_dense_code<S, O>(
        esn: &EchoStateNetwork<S, DenseReservoir<S>, O>,
    ) -> Result<String, String>
    where
        S: RealScalar + std::fmt::Display,
        O: reservoir_core::Readout<S> + GetWeights<S>,
    {
        let w_in = &esn.reservoir.w_in;
        let w = &esn.reservoir.w;
        let w_out = esn.readout.weights();

        let input_dim = esn.reservoir.input_dim;
        let reservoir_size = esn.reservoir.res_state.len();
        let output_dim = esn.readout.output_dim();
        let ext_size = 1 + input_dim + reservoir_size;
        let leaking_rate = esn.reservoir.leaking_rate;
        let initial_state = &esn.reservoir.res_state;

        // Column-major flattening, matching `nalgebra`'s `from_column_slice`.
        let fmt_scalar = |v: &[S]| -> String {
            v.iter()
                .map(|x| format!("{:.8}", x))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let w_in_flat: Vec<S> = w_in.iter().cloned().collect();
        let w_flat: Vec<S> = w.iter().cloned().collect();
        let w_out_flat: Vec<S> = w_out.iter().cloned().collect();
        let state_flat: Vec<S> = initial_state.iter().cloned().collect();

        let mut code = String::new();

        writeln!(
            code,
            "// Auto-generated by reservoir-train::codegen (dense)"
        )
        .unwrap();
        // Generated module: consumers may not use every emitted const.
        writeln!(code, "#![allow(dead_code)]").unwrap();
        writeln!(
            code,
            "use reservoir_infer::{{StaticReservoir, StaticReadout, StaticESN}};"
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

        writeln!(
            code,
            "// W_in: shape (RESERVOIR_SIZE, INPUT_DIM), column-major",
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_IN_DATA: [f32; {}] = [{}];",
            w_in_flat.len(),
            fmt_scalar(&w_in_flat)
        )
        .unwrap();
        writeln!(
            code,
            "// W: shape (RESERVOIR_SIZE, RESERVOIR_SIZE), column-major",
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_DATA: [f32; {}] = [{}];",
            w_flat.len(),
            fmt_scalar(&w_flat)
        )
        .unwrap();
        writeln!(
            code,
            "// W_out: shape (OUTPUT_DIM, EXTENDED_SIZE), column-major",
        )
        .unwrap();
        writeln!(
            code,
            "pub const W_OUT_DATA: [f32; {}] = [{}];",
            w_out_flat.len(),
            fmt_scalar(&w_out_flat)
        )
        .unwrap();
        writeln!(
            code,
            "pub const INITIAL_STATE_DATA: [f32; {}] = [{}];",
            state_flat.len(),
            fmt_scalar(&state_flat)
        )
        .unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "pub type Reservoir = StaticReservoir<f32, INPUT_DIM, RESERVOIR_SIZE, EXTENDED_SIZE>;"
        )
        .unwrap();
        writeln!(
            code,
            "pub type Readout = StaticReadout<f32, EXTENDED_SIZE, OUTPUT_DIM>;"
        )
        .unwrap();
        writeln!(code, "pub type Model = StaticESN<f32, Reservoir, Readout>;").unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "/// Reconstruct the trained model using only `reservoir-infer`."
        )
        .unwrap();
        writeln!(code, "pub fn build() -> Model {{").unwrap();
        writeln!(
            code,
            "    let w_in = SMatrix::<f32, RESERVOIR_SIZE, INPUT_DIM>::from_column_slice(&W_IN_DATA);"
        )
        .unwrap();
        writeln!(
            code,
            "    let w = SMatrix::<f32, RESERVOIR_SIZE, RESERVOIR_SIZE>::from_column_slice(&W_DATA);"
        )
        .unwrap();
        writeln!(
            code,
            "    let mut reservoir = Reservoir::create(w_in, w, LEAKING_RATE);"
        )
        .unwrap();
        writeln!(
            code,
            "    reservoir.res_state = SVector::<f32, RESERVOIR_SIZE>::from_column_slice(&INITIAL_STATE_DATA);"
        )
        .unwrap();
        writeln!(
            code,
            "    let w_out = SMatrix::<f32, OUTPUT_DIM, EXTENDED_SIZE>::from_column_slice(&W_OUT_DATA);"
        )
        .unwrap();
        writeln!(code, "    let readout = Readout::create(w_out);").unwrap();
        writeln!(code, "    StaticESN::new(reservoir, readout)").unwrap();
        writeln!(code, "}}").unwrap();
        writeln!(code).unwrap();

        writeln!(
            code,
            "/// Advance the model by one step (hides the const-generic dimensions)."
        )
        .unwrap();
        writeln!(
            code,
            "pub fn step(model: &mut Model, x: &SVector<f32, INPUT_DIM>) -> SVector<f32, OUTPUT_DIM> {{"
        )
        .unwrap();
        writeln!(
            code,
            "    model.predict::<INPUT_DIM, OUTPUT_DIM, EXTENDED_SIZE>(x)"
        )
        .unwrap();
        writeln!(code, "}}").unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ESNBuilder;
    use nalgebra::{SMatrix, SVector};
    use reservoir_infer::reservoir::{StaticCsrMatrix, StaticSparseReservoir};
    use reservoir_infer::{StaticReadout, StaticReservoir};

    // The generated code rebuilds a `StaticReservoir` / `StaticReadout` from the
    // dynamic `DenseReservoir` weights, flattened column-major. This asserts that
    // convention is correct: the static forward pass must match the dynamic one.
    #[test]
    fn test_generate_dense_code_matches_dynamic_forward_pass() {
        const IN: usize = 1;
        const N: usize = 4;
        const EXT: usize = 1 + IN + N; // 6
        const OUT: usize = 1;

        let mut esn = ESNBuilder::<f32>::new(IN, OUT)
            .units(N)
            .spectral_radius(0.9)
            .leaking_rate(0.3)
            .seed(7)
            .build();

        // Reconstruct the static model exactly as `generate_dense_code` serializes:
        // column-major flatten -> `from_column_slice`.
        let w_in_flat: Vec<f32> = esn.reservoir.w_in.iter().cloned().collect();
        let w_flat: Vec<f32> = esn.reservoir.w.iter().cloned().collect();
        let w_out_flat: Vec<f32> = esn.readout.weights().iter().cloned().collect();

        let w_in = SMatrix::<f32, N, IN>::from_column_slice(&w_in_flat);
        let w = SMatrix::<f32, N, N>::from_column_slice(&w_flat);
        let mut static_res =
            StaticReservoir::<f32, IN, N, EXT>::create(w_in, w, esn.reservoir.leaking_rate);
        let w_out = SMatrix::<f32, OUT, EXT>::from_column_slice(&w_out_flat);
        let static_readout = StaticReadout::<f32, EXT, OUT>::create(w_out);

        // Both models start from a zero state and advance in lockstep.
        for &u in &[0.5f32, -0.3, 0.1, 0.9, -0.7] {
            let dyn_out = esn.predict(alloc::vec![u])[0];
            let ext = static_res.step(&SVector::<f32, IN>::new(u));
            let stat_out = static_readout.predict(ext)[0];
            assert!(
                (dyn_out - stat_out).abs() < 1e-4,
                "u={u}: dynamic={dyn_out} static={stat_out}"
            );
        }
    }

    #[test]
    fn test_generate_dense_code_smoke() {
        let esn = ESNBuilder::<f32>::new(1, 1).units(3).seed(1).build();
        let code = StaticModelGenerator::generate_dense_code(&esn).unwrap();

        assert!(code.contains("pub const INPUT_DIM: usize = 1;"));
        assert!(code.contains("pub const RESERVOIR_SIZE: usize = 3;"));
        assert!(code.contains("pub const OUTPUT_DIM: usize = 1;"));
        assert!(code.contains("pub const EXTENDED_SIZE: usize = 5;"));
        assert!(code.contains("pub const W_IN_DATA: [f32; 3]"));
        assert!(code.contains("pub const W_DATA: [f32; 9]"));
        assert!(code.contains("pub const W_OUT_DATA: [f32; 5]"));
        assert!(code.contains("pub fn build() -> Model {"));
        assert!(code.contains("pub fn step("));
        assert!(code.contains("StaticESN::new(reservoir, readout)"));
    }

    // Same parity guarantee for the sparse (CSR) generator: the static forward pass
    // reconstructed the codegen way must match the dynamic `SparseReservoir`.
    #[test]
    fn test_generate_sparse_code_matches_dynamic_forward_pass() {
        const IN: usize = 1;
        const N: usize = 4;
        const EXT: usize = 1 + IN + N; // 6
        const OUT: usize = 1;

        let mut esn = ESNBuilder::<f32>::new(IN, OUT)
            .units(N)
            .connectivity(2)
            .input_connectivity(1)
            .spectral_radius(0.9)
            .leaking_rate(0.3)
            .seed(11)
            .build_sparse();

        // Snapshot the trained weights into owned buffers exactly as the generator
        // serializes them (CSR indices usize -> u16), releasing the borrow on `esn`.
        let w_row_ptr: Vec<u16> = esn.reservoir.w.row_ptr.iter().map(|&x| x as u16).collect();
        let w_col_idx: Vec<u16> = esn.reservoir.w.col_idx.iter().map(|&x| x as u16).collect();
        let w_values: Vec<f32> = esn.reservoir.w.values.clone();
        let win_row_ptr: Vec<u16> = esn
            .reservoir
            .w_in
            .row_ptr
            .iter()
            .map(|&x| x as u16)
            .collect();
        let win_col_idx: Vec<u16> = esn
            .reservoir
            .w_in
            .col_idx
            .iter()
            .map(|&x| x as u16)
            .collect();
        let win_values: Vec<f32> = esn.reservoir.w_in.values.clone();
        let w_out_flat: Vec<f32> = esn.readout.weights().iter().cloned().collect();
        let leaking = esn.reservoir.leaking_rate;

        let s_w_in = StaticCsrMatrix::<f32, N, IN>::new(&win_row_ptr, &win_col_idx, &win_values);
        let s_w = StaticCsrMatrix::<f32, N, N>::new(&w_row_ptr, &w_col_idx, &w_values);
        let mut static_res = StaticSparseReservoir::<f32, IN, N, EXT>::create(s_w_in, s_w, leaking);
        let s_w_out = SMatrix::<f32, OUT, EXT>::from_column_slice(&w_out_flat);
        let static_readout = StaticReadout::<f32, EXT, OUT>::create(s_w_out);

        for &u in &[0.5f32, -0.3, 0.1, 0.9, -0.7] {
            let dyn_out = esn.predict(alloc::vec![u])[0];
            let ext = static_res.step(&SVector::<f32, IN>::new(u));
            let stat_out = static_readout.predict(ext)[0];
            assert!(
                (dyn_out - stat_out).abs() < 1e-4,
                "u={u}: dynamic={dyn_out} static={stat_out}"
            );
        }
    }

    #[test]
    fn test_generate_sparse_code_emits_build_helpers() {
        let esn = ESNBuilder::<f32>::new(1, 1)
            .units(4)
            .connectivity(2)
            .input_connectivity(1)
            .seed(3)
            .build_sparse();
        let code = StaticModelGenerator::generate_sparse_code(&esn).unwrap();

        assert!(code.contains("pub type Model = StaticESN"));
        assert!(code.contains("pub fn build() -> Model {"));
        assert!(code.contains("pub fn step("));
        assert!(code.contains("StaticESN::new(reservoir, readout)"));
        assert!(code.contains("use reservoir_infer::esn::StaticESN;"));
    }
}
