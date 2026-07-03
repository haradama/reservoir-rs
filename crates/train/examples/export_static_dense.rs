//! Train a dense ESN, then export it as a self-contained static-inference module.
//!
//! This demonstrates the **train -> infer-only** hand-off:
//! 1. Build and train a dense ESN with `reservoir-train`.
//! 2. Emit Rust source with [`StaticModelGenerator::generate_dense_code`].
//! 3. The generated module depends only on `reservoir-infer` (+ `nalgebra`) and
//!    exposes `build()` / `step()` for `no_std`-friendly static inference.
//!
//! The generated code is written to **stdout**; diagnostics go to **stderr**, so
//! you can capture just the module:
//!
//! ```bash
//! cargo run -p reservoir-train --example export_static_dense > model.rs
//! ```
//!
//! Then, from an inference crate that depends only on `reservoir-infer`:
//!
//! ```rust,ignore
//! mod model; // the generated file
//! use nalgebra::SVector;
//!
//! let mut esn = model::build();
//! let y = model::step(&mut esn, &SVector::<f32, { model::INPUT_DIM }>::new(0.5));
//! ```

use reservoir_train::{ESNBuilder, ESNFitRidge, StaticModelGenerator};

fn main() {
    type S = f32;

    // --- toy time-series: u_t = sin(0.02 t), predict u_{t+1} ---
    let steps = 1500usize;
    let series: Vec<S> = (0..=steps).map(|t| ((t as f32) * 0.02).sin()).collect();
    let inputs: Vec<Vec<S>> = series[..steps].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<S>> = series[1..].iter().map(|&v| vec![v]).collect();

    // --- build + train a dense ESN ---
    let mut esn = ESNBuilder::<S>::new(1, 1)
        .units(16)
        .spectral_radius(0.9)
        .leaking_rate(0.3)
        .input_scaling(0.5)
        .seed(42)
        .build();
    esn.fit(&inputs, &targets, 1e-6, 50);

    // --- export as a static, infer-only module (captures the post-training state) ---
    // Do this BEFORE running any prediction: `generate_dense_code` embeds the current
    // reservoir state as `INITIAL_STATE_DATA`, and `predict` would advance it.
    let code = StaticModelGenerator::generate_dense_code(&esn)
        .expect("failed to generate static dense model");

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
