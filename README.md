# reservoir-rs

A minimal, experimental Rust workspace for reservoir computing / Echo State Networks (ESNs).

This project focuses on:

- reusable **core traits** (reservoir / readout / trainer)
- optional **no_std** support via feature flags
- simple **datasets** for time-series experiments
- lightweight **training + inference** components

> Note: This is early-stage software. APIs, module layout, and crate names may change.

## What’s inside (high level)

- **Core**: traits, scalar/type aliases, basic metrics (MSE/RMSE/NRMSE/R²)
- **Inference**: ESN-style reservoir + readout building blocks (dynamic and/or const-generic variants depending on features)
- **Training**: simple readout training (e.g., ridge / lasso-style solvers)
- **Datasets**: synthetic time-series generators (for quick experiments)
- **Embedded/QEMU tests**: optional no_std inference checks on emulated targets

## Build

This is a Cargo workspace.

```bash
cargo build
cargo test
````

Examples (if available in the workspace):

```bash
cargo run --example <name>
```

## no_std / features

Some components can be built without the standard library.

- Use `default-features = false` when you want `no_std`.
- Enable `alloc` when you need heap-backed vectors/matrices.
- Enable `libm` when targeting platforms without `std` floating-point support.

Exact feature sets may change as the project evolves.

## License

Apache License 2.0
