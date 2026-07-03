//! Train a sparse (CSR) ESN, then export it as a self-contained static module.
//!
//! Sparse counterpart of `export_static_dense`. It demonstrates the same
//! **train -> infer-only** hand-off, but the reservoir is stored as CSR matrices,
//! which is usually the better fit for larger `no_std` / embedded models.
//!
//! The generated code is written to **stdout**; diagnostics go to **stderr**:
//!
//! ```bash
//! cargo run -p reservoir-train --example export_static_sparse > model.rs
//! ```
//!
//! Then, from an inference crate that depends only on `reservoir-infer`:
//!
//! ```rust,ignore
//! mod model; // the generated file
//! use nalgebra::SVector;
//!
//! let mut esn = model::build();
//! let y = model::step(&mut esn, &SVector::<f32, 1>::new(0.5));
//! ```

use reservoir_train::{ESNBuilder, ESNFitRidge, StaticModelGenerator};

fn main() {
    type S = f32;

    // --- toy time-series: u_t = sin(0.02 t), predict u_{t+1} ---
    let steps = 1500usize;
    let series: Vec<S> = (0..=steps).map(|t| ((t as f32) * 0.02).sin()).collect();
    let inputs: Vec<Vec<S>> = series[..steps].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<S>> = series[1..].iter().map(|&v| vec![v]).collect();

    // --- build + train a sparse ESN ---
    let mut esn = ESNBuilder::<S>::new(1, 1)
        .units(32)
        .connectivity(4)
        .input_connectivity(1)
        .spectral_radius(0.9)
        .leaking_rate(0.3)
        .input_scaling(0.5)
        .seed(42)
        .build_sparse();
    esn.fit(&inputs, &targets, 1e-6, 50);

    // --- export as a static, infer-only module (captures the post-training state) ---
    // Do this BEFORE running any prediction: the generator embeds the current
    // reservoir state as `INITIAL_STATE_DATA`, and `predict` would advance it.
    let code = StaticModelGenerator::generate_sparse_code(&esn)
        .expect("failed to generate static sparse model");

    // --- reference predictions from the trained (dynamic) model ---
    // These start from the same state the generated `build()` restores, so the
    // infer-only consumer must reproduce them exactly.
    let sample_inputs = [0.5f32, -0.3, 0.1, 0.9, -0.7];
    eprintln!("# reference predictions from the trained dynamic model:");
    for &u in &sample_inputs {
        let y = esn.predict(vec![u])[0];
        eprintln!("#   u = {u:+.3}  ->  y = {y:+.6}");
    }
    eprintln!(
        "# reservoir units = {}, extended state = {}",
        esn.reservoir.res_state.len(),
        esn.reservoir.ext_state.len()
    );

    // Generated module -> stdout.
    print!("{code}");
}
